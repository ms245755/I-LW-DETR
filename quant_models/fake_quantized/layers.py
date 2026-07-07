import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=10)

from .utils import *
from util.decorators import *
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple

enable_assert = False
enable_casting = False


@track_init_args
class IViTLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer

    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_bit=8,
        bias_bit=32,
        per_channel=True,
        quant_mode="symmetric",
    ):
        super(IViTLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.quant_mode = quant_mode

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        self.register_buffer("fc_scaling_factor", nn.Parameter(torch.ones(self.out_features)))
        self.register_buffer("weight_integer", nn.Parameter(torch.ones_like(self.weight)))
        if self.bias is not None:
            self.register_buffer("bias_integer", nn.Parameter(torch.ones_like(self.bias)))

    def __repr__(self):
        s = super(IViTLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, act_scaling_factor=None):
        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception("For weight, we only support per_channel quantization.")

            self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True
        )

        bias_scaling_factor = self.fc_scaling_factor * act_scaling_factor

        if self.bias is not None and self.quantize_bias:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True
            )
        else:
            self.bias_integer = None

        act_scaling_factor = act_scaling_factor.view(1, -1)
        x = round_ste.apply(x / act_scaling_factor)

        x_out = F.linear(x, weight=self.weight_integer, bias=self.bias_integer)
        x_out = round_ste.apply(x_out)

        x_out *= bias_scaling_factor
        if self.bias is not None and not self.quantize_bias:
            x_out += self.bias

        return x_out, bias_scaling_factor


@track_init_args
class IViTAct(nn.Module):
    """
    Class to quantize given activations. Probably the most confusing class.
    Here is where we apply symmetric uniform quantization to an input vector x.
    For more details, like how the EMA works, etc, look at the I-Segmenter paper.
    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'asymmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    observer_mode : 'ema_minmax', 'percentile', or 'mean', default 'ema_minmax'
        Activation range observer used while running statistics are enabled.
        'mean' uses min/max observations with equal-weight averaging across batches.
    """

    _default_observer_mode = "ema_minmax"
    _default_percentile = 99.9
    _default_use_dyadic_scale = False
    _default_dyadic_scale_bits = 31

    @classmethod
    def configure_defaults(
        cls,
        observer_mode=None,
        percentile=None,
        use_dyadic_scale=None,
        dyadic_scale_bits=None,
    ):
        if observer_mode is not None:
            cls._default_observer_mode = observer_mode
        if percentile is not None:
            cls._default_percentile = percentile
        if use_dyadic_scale is not None:
            cls._default_use_dyadic_scale = use_dyadic_scale
        if dyadic_scale_bits is not None:
            cls._default_dyadic_scale_bits = dyadic_scale_bits

    def __init__(
        self,
        activation_bit=8,
        act_range_momentum=0.99,
        running_stat=True,
        per_channel=False,
        quant_mode="symmetric",
        observer_mode=None,
        percentile=None,
        use_dyadic_scale=True,
        dyadic_scale_bits=16,
    ):
        super(IViTAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.observer_mode = observer_mode or self._default_observer_mode
        self.percentile = self._default_percentile if percentile is None else percentile
        self.use_dyadic_scale = (
            self._default_use_dyadic_scale
            if use_dyadic_scale is None
            else use_dyadic_scale
        )
        self.dyadic_scale_bits = (
            self._default_dyadic_scale_bits
            if dyadic_scale_bits is None
            else dyadic_scale_bits
        )

        self.register_buffer("act_scaling_factor", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("out_scaling_factor", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("act_scale_mantissa", torch.ones(1, dtype=torch.int32))
        self.register_buffer("act_scale_shift", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("min_val", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("max_val", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("n_batches", torch.tensor(0, dtype=torch.long))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        min_val = self.min_val.min().item() if self.min_val.numel() > 1 else self.min_val.item()
        max_val = self.max_val.max().item() if self.max_val.numel() > 1 else self.max_val.item()
        return (
            "{0}(activation_bit={1}, "
            "quant_mode: {2}, observer_mode: {3}, per_channel: {4}, "
            "Act_min: {5:.2f}, Act_max: {6:.2f})".format(
                self.__class__.__name__,
                self.activation_bit,
                self.quant_mode,
                self.observer_mode,
                self.per_channel,
                min_val,
                max_val,
            )
        )

    def _snap_scale_to_dyadic(self, scale):
        mantissa, shift = batch_frexp(scale.detach().float(), max_bit=self.dyadic_scale_bits)
        dyadic_scale = mantissa.to(device=scale.device, dtype=scale.dtype) / (
            2.0 ** shift.to(device=scale.device, dtype=scale.dtype)
        )
        dyadic_scale = dyadic_scale.view_as(scale)

        self.act_scale_mantissa = mantissa.to(device=scale.device, dtype=torch.int32)
        self.act_scale_shift = shift.to(device=scale.device, dtype=torch.int32)

        return dyadic_scale

    def set_dyadic_scale(self, mantissa, shift):
        mantissa = torch.as_tensor(mantissa, device=self.act_scaling_factor.device)
        shift = torch.as_tensor(shift, device=self.act_scaling_factor.device)

        self.act_scale_mantissa = mantissa.to(dtype=torch.int32)
        self.act_scale_shift = shift.to(dtype=torch.int32)
        self.act_scaling_factor = self.act_scale_mantissa.to(
            dtype=self.act_scaling_factor.dtype
        ) / (2.0 ** self.act_scale_shift.to(dtype=self.act_scaling_factor.dtype))
        self.use_dyadic_scale = True

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def _set_min_param(self, new_tensor):
        """Method to reassign the underlying data of the parameter."""
        self.min_val.data = new_tensor.clone().min()

    def _set_max_param(self, new_tensor):
        """Method to reassign the underlying data of the parameter."""
        self.max_val.data = new_tensor.clone().max()

    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def _broadcast_scale(self, scale, x):
        if scale.numel() == 1:
            return scale
        if x.dim() == 2:
            return scale.view(1, -1)
        if x.dim() == 3:
            return scale.view(1, 1, -1)
        if x.dim() == 4:
            return scale.view(1, -1, 1, 1)
        if x.dim() == 1:
            return scale
        raise NotImplementedError

    def forward(self, x, act_scaling_factor=None, x_2=None, act_scaling_factor_2=None):
        # collect runnng stats
        with torch.no_grad():
            x_act = x if x_2 is None else x_2 + x
            collect_current_stat = self.running_stat or self.n_batches.item() == 0
            if collect_current_stat:
                if len(x_act.shape) == 4:
                    x_act = x_act.permute(0, 2, 3, 1)
                v = x_act.reshape(-1, x_act.shape[-1])
                v = v.transpose(0, 1)

                if self.observer_mode == "percentile":
                    upper_q = min(max(self.percentile / 100.0, 0.0), 1.0)
                    lower_q = min(max(1.0 - upper_q, 0.0), 1.0)
                    v_float = v.float()
                    cur_min = torch.quantile(v_float, lower_q, dim=1).to(v.dtype)
                    cur_max = torch.quantile(v_float, upper_q, dim=1).to(v.dtype)
                elif self.observer_mode in ("ema_minmax", "mean"):
                    cur_min = v.min(dim=1).values
                    cur_max = v.max(dim=1).values
                else:
                    raise ValueError("unknown observer_mode: {}".format(self.observer_mode))
                if not self.per_channel:
                    cur_min = cur_min.min()
                    cur_max = cur_max.max()
                if self.n_batches.item() == 0:
                    self.min_val = cur_min
                    self.max_val = cur_max
                elif self.observer_mode == "mean":
                    batch_count = self.n_batches.to(dtype=cur_min.dtype, device=cur_min.device) + 1
                    self.min_val = self.min_val + (cur_min - self.min_val) / batch_count
                    self.max_val = self.max_val + (cur_max - self.max_val) / batch_count
                else:
                    self.min_val = self.min_val * self.act_range_momentum + cur_min * (
                        1 - self.act_range_momentum
                    )
                    self.max_val = self.max_val * self.act_range_momentum + cur_max * (
                        1 - self.act_range_momentum
                    )
                if self.running_stat:
                    self.n_batches += 1

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val
            )
            if self.use_dyadic_scale:
                self.act_scaling_factor = self._snap_scale_to_dyadic(self.act_scaling_factor)

        if act_scaling_factor is None:
            # this is for the x quantization
            quant_act_int = self.act_function(
                x, self.activation_bit, self.act_scaling_factor, False
            )
        else:
            quant_act_int = fixedpoint_mul.apply(
                x,
                act_scaling_factor,
                self.activation_bit,
                self.quant_mode,
                self.act_scaling_factor,
                x_2,
                act_scaling_factor_2,
            )

        correct_output_scale = self.act_scaling_factor.view(-1)
        output_scale = self._broadcast_scale(correct_output_scale, x)

        return quant_act_int * output_scale, correct_output_scale


@track_init_args
class IViTMatMul(nn.Module):
    """
    Class to quantize weights of given matmul layer
    """

    def __init__(self):
        super(IViTMatMul, self).__init__()
        self.register_buffer("act_scaling_factor", nn.Parameter(torch.tensor(1.0)))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A = round_ste.apply(A / pre_act_scaling_factor_A)
        B = round_ste.apply(B / pre_act_scaling_factor_B)
        # A = A / pre_act_scaling_factor_A
        # B = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        out = A @ B
        out *= act_scaling_factor

        return out, act_scaling_factor


@track_init_args
class IViTConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        weight_bit=8,
        bias_bit=32,
        quant_mode="symmetric",
        per_channel=True,
        weight_percentile=0,
    ):
        super(IViTConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True

        self.register_buffer("conv_scaling_factor", nn.Parameter(torch.zeros(self.out_channels)))
        self.register_buffer("weight_integer", nn.Parameter(torch.zeros_like(self.weight)))
        self.register_buffer("bias_integer", nn.Parameter(torch.zeros_like(self.bias)))

    def __repr__(self):
        s = super(IViTConv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def reshape_and_quantize(self, x, act_scaling_factor=None):
        x = x.flatten(2).transpose(1, 2)
        return self.post_act_scaler(x, act_scaling_factor)

    def quantized_weight(self):
        if self.quant_mode == "symmetric":
            weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(dim=1).values
                cur_max = v.max(dim=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception("For weight, we only support per_channel quantization.")

            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True
        )
        return self.weight_integer, self.conv_scaling_factor

    def forward(self, x, act_scaling_factor=None):
        self.weight_integer, self.conv_scaling_factor = self.quantized_weight()
        bias_scaling_factor = self.conv_scaling_factor * act_scaling_factor
        self.bias_integer = SymmetricQuantFunction.apply(
            self.bias, self.bias_bit, bias_scaling_factor, True
        )

        act_scaling_factor = act_scaling_factor.view(1, -1, 1, 1)
        x = round_ste.apply(x / act_scaling_factor)
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        x_out = F.conv2d(
            x,
            self.weight_integer,
            self.bias_integer,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        x_out = x_out * correct_output_scale

        return x_out, correct_output_scale
    
@track_init_args
class IViTConvTranspose2d(nn.ConvTranspose2d):
    """
    Class to quantize weights of given transpose convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 8
        Bitwidth for quantized weights.
    bias_bit : int, default 32
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default True
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        weight_bit=8,
        bias_bit=32,
        quant_mode="symmetric",
        per_channel=True,
        weight_percentile=0,
    ):
        super(IViTConvTranspose2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True

        self.register_buffer("conv_scaling_factor", nn.Parameter(torch.zeros(self.out_channels)))
        self.register_buffer("weight_integer", nn.Parameter(torch.zeros_like(self.weight)))
        self.register_buffer("bias_integer", nn.Parameter(torch.zeros_like(self.bias)))

    def __repr__(self):
        s = super(IViTConvTranspose2d, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def reshape_and_quantize(self, x, act_scaling_factor=None):
        x = x.flatten(2).transpose(1, 2)
        return self.post_act_scaler(x, act_scaling_factor)

    def forward(self, x, act_scaling_factor=None):
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(dim=1).values
                cur_max = v.max(dim=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception("For weight, we only support per_channel quantization.")

            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True
        )
        bias_scaling_factor = self.conv_scaling_factor * act_scaling_factor
        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, bias_scaling_factor, True
        )

        act_scaling_factor = act_scaling_factor.view(1, -1, 1, 1)
        x = round_ste.apply(x / act_scaling_factor)
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        x_out = F.conv_transpose2d(
            x,
            self.weight_integer,
            self.bias_integer,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        x_out = x_out * correct_output_scale

        return x_out, correct_output_scale


@track_init_args
class IViTLayerNorm(nn.LayerNorm):
    """
    Implementation of I-LayerNorm
    Class to quantize given LayerNorm layer
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        elementwise_affine=True,
        observe_stat=False,
        block_name="",
        dim=-1,
    ):
        super(IViTLayerNorm, self).__init__(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )
        self.dim = dim
        norm_scale_shape = (1, normalized_shape, 1, 1) if dim == 1 else (normalized_shape,)
        self.m = 30
        self.register_buffer("norm_scaling_factor", nn.Parameter(torch.ones(*norm_scale_shape)))
        self.register_buffer("bias_integer", nn.Parameter(torch.ones(*norm_scale_shape)))
        self.register_buffer(
            "base_act_scaling_factor",
            nn.Parameter(torch.sqrt(torch.tensor(normalized_shape, dtype=torch.float)) / 2.0**self.m),
        )
        self.block_name = block_name
        self.observe_stat = observe_stat

    def fix(self):
        pass

    def unfix(self):
        pass

    def _normalized_dim(self, x):
        dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        if dim < 0 or dim >= x.dim():
            raise ValueError("invalid LayerNorm dim {} for input with {} dims".format(self.dim, x.dim()))
        if x.shape[dim] != self.normalized_shape[0]:
            raise ValueError(
                "LayerNorm normalized_shape {} does not match input shape {} at dim {}".format(
                    self.normalized_shape[0], tuple(x.shape), dim
                )
            )
        return dim

    def _view_affine(self, x, dim):
        if dim == x.dim() - 1:
            return self.weight, self.bias, self.base_act_scaling_factor * self.weight

        view_shape = [1] * x.dim()
        view_shape[dim] = self.normalized_shape[0]
        weight = self.weight.view(view_shape)
        bias = self.bias.view(view_shape)
        act_scaling_factor = (self.base_act_scaling_factor * self.weight).view(view_shape)
        return weight, bias, act_scaling_factor

    def forward(self, x, act_scaling_factor=None):
        # Normalization: computes mean and variance(std)
        dim = self._normalized_dim(x)
        # if act_scaling_factor is None:
        #     with torch.no_grad():
        #         target = 2.0 ** 7 - 1
        #         max_abs = x.detach().abs().amax(dim=dim, keepdim=True)
        #         act_scaling_factor = torch.clamp(max_abs / target, min=torch.finfo(x.dtype).eps)
        x_int = round_ste.apply(x / act_scaling_factor)
        mean_int = round_ste.apply(x_int.sum(dim=dim, keepdim=True) / x_int.shape[dim])

        y_int = x_int - mean_int
        y_sq_int = y_int**2
        var_int = torch.sum(y_sq_int, axis=dim, keepdim=True)

        # Integer Iteration
        k = 2.0**16
        for _ in range(20):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)
            k = k_1
        std_int = k
        
        exp = floor_ste.apply((2.0**(self.m+1) - 1) / std_int)
        factor = exp
        y_int = floor_ste.apply(y_int * factor / 2)
        # scaling and shifting
        weight, bias, act_scaling_factor = self._view_affine(x, dim)
        bias = bias / weight
        bias_int = floor_ste.apply(bias / self.base_act_scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int
        x = y_int * act_scaling_factor
        self.norm_scaling_factor = act_scaling_factor

        save_tensor_from_pathname(x, self.block_name+"output", observe_stat=self.observe_stat)
        return x, act_scaling_factor
    
@track_init_args
class IViTLayerNormChannelFirst(IViTLayerNorm):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super(IViTLayerNormChannelFirst, self).__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            dim=1,
        )
    
class IViTGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8, n = 23):
        super(IViTGELU, self).__init__()
        self.output_bit = output_bit

        self.n = n  # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer("act_scaling_factor", torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2**4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2.0 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2.0**self.n

        return exp_int, scaling_factor

    def forward(self, x, act_scaling_factor=None):
        pre_x_int = x / act_scaling_factor
        act_scaling_factor_sig = act_scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, act_scaling_factor_sig)  # e^(x-x_max)

        exp_int_max, _ = self.int_exp_shift(-x_int_max, act_scaling_factor_sig)  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2.0**31 - 1)
        factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(exp_int * factor / 2.0 ** (31 - self.output_bit + 1))
        sigmoid_act_scaling_factor = torch.Tensor([1 / 2.0 ** (self.output_bit - 1)]).to(x.device)

        x_int = pre_x_int * sigmoid_int
        act_scaling_factor = act_scaling_factor * sigmoid_act_scaling_factor
        self.act_scaling_factor = act_scaling_factor
        return x_int * act_scaling_factor, act_scaling_factor

@track_init_args
class IViTGELUWithSigmoidRange(nn.Module):
    """
    IViTGELU with the sigmoid input range transform from Eq. (15)-(16).

    The transform is applied only to the sigmoid approximation input:
        I_X' = clamp(I_X * 2^M, -127, 127)
        S_X' = S_X / 2^M
    where S_X is the GELU sigmoid scale, i.e. the input scale multiplied by 1.702.
    """

    def __init__(self, output_bit=8, input_bit=8, sigmoid_bound=6.0):
        super(IViTGELUWithSigmoidRange, self).__init__()
        self.output_bit = output_bit
        self.input_bit = input_bit
        self.sigmoid_bound = sigmoid_bound

        self.n = 23  # sufficiently large integer

        self.register_buffer("act_scaling_factor", torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def _sigmoid_range_transform(self, x_int, scaling_factor):
        target_int = 2.0 ** (self.input_bit - 1) - 1
        target_scaling_factor = torch.tensor(
            self.sigmoid_bound / target_int,
            device=scaling_factor.device,
            dtype=scaling_factor.dtype,
        )

        with torch.no_grad():
            ratio = scaling_factor / target_scaling_factor
            m = torch.clamp(torch.ceil(torch.log2(torch.clamp(ratio, min=1.0))) - 1.0, min=0.0)
            shift_factor = 2.0 ** m

        x_int = torch.clamp(x_int * shift_factor, min=-target_int, max=target_int)
        scaling_factor = scaling_factor / shift_factor

        return x_int, scaling_factor

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2**4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2.0 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2.0**self.n

        return exp_int, scaling_factor

    def forward(self, x, act_scaling_factor=None):
        pre_x_int = x / act_scaling_factor
        act_scaling_factor_sig = act_scaling_factor * 1.702

        sigmoid_x_int, sigmoid_scaling_factor = self._sigmoid_range_transform(
            pre_x_int,
            act_scaling_factor_sig,
        )

        x_int_max, _ = sigmoid_x_int.max(dim=-1, keepdim=True)
        x_int = sigmoid_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, sigmoid_scaling_factor)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, sigmoid_scaling_factor)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum = torch.clamp(exp_int_sum, min=1, max=2.0**31 - 1)
        factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(exp_int * factor / 2.0 ** (31 - self.output_bit + 1))
        sigmoid_act_scaling_factor = torch.Tensor([1 / 2.0 ** (self.output_bit - 1)]).to(x.device)

        x_int = pre_x_int * sigmoid_int
        act_scaling_factor = act_scaling_factor * sigmoid_act_scaling_factor
        self.act_scaling_factor = act_scaling_factor
        return x_int * act_scaling_factor, act_scaling_factor

@track_init_args
class IViTSiLU(nn.Module):
    """
    Integer-ish SiLU (Swish-1): SiLU(x) = x * sigmoid(x)

    Matches the structure of IViTGELU, but removes the 1.702 slope.
    Uses the same exp-shift trick + max-subtraction for a stable integer sigmoid.
    """

    def __init__(self, output_bit=8, n=23, lambda_scaler=6):
        super().__init__()
        self.output_bit = output_bit

        self.n = n  # sufficiently large integer
        self.lambda_scaler = lambda_scaler
        self.m = 31
        self.register_buffer("act_scaling_factor", torch.zeros(1))
        self.register_buffer(
            "sigmoid_scaling_factor",
            torch.Tensor([1 / 2.0 ** (self.output_bit - 1)]),
        )

    def set_calibration_params(self, n=None, lambda_scaler=None, output_bit=None):
        if output_bit is not None and output_bit != self.output_bit:
            self.output_bit = output_bit
            self.sigmoid_scaling_factor.data = torch.tensor(
                [1 / 2.0 ** (self.output_bit - 1)],
                device=self.sigmoid_scaling_factor.device,
                dtype=self.sigmoid_scaling_factor.dtype,
            )
        if n is not None:
            self.n = n
        if lambda_scaler is not None:
            self.lambda_scaler = lambda_scaler

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        scaled_floor = floor_ste.apply(x_int / 2)
        scaled_exp_floor = floor_ste.apply(x_int / (2.0**4))
        x_int = x_int + scaled_floor - scaled_exp_floor

        with torch.no_grad():
            x0_int = torch.ceil(1.0 / scaling_factor)

        min_x_int = torch.clamp(self.n * self.lambda_scaler * (-x0_int), min=-(2.0**self.m))
        x_int = torch.maximum(x_int, min_x_int)

        q = floor_ste.apply(x_int / (-x0_int))
        r = -(x_int - q * (-x0_int))

        exp_int = floor_ste.apply((-r) / 2)
        exp_int += x0_int
        exp_int = floor_ste.apply(exp_int * 2.0 ** (self.n - q))

        scaling_factor = scaling_factor / 2.0**self.n
        return exp_int, scaling_factor

    def forward(self, x, act_scaling_factor=None):
        pre_x_int = round_ste.apply(x / act_scaling_factor)

        # For SiLU, sigmoid input scale is just act_scaling_factor (no 1.702 multiplier)
        scaling_factor_sig = act_scaling_factor

        # A = 2.0
        # A_int = round_ste.apply(
        #     torch.tensor(A, device=x.device, dtype=x.dtype) / act_scaling_factor
        # )

        # pos_mask = pre_x_int > 0

        # x_int_max_base = torch.clamp(pre_x_int, min=0)

        # x_int_max = torch.where(
        #     pos_mask,
        #     pre_x_int + A_int,
        #     x_int_max_base
        # )


        x_int_max = torch.clamp(pre_x_int, min=0)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)
        exp_int_sum = exp_int + exp_int_max

        # Clamp to avoid overflow or divide-by-zero
        exp_int_sum = torch.clamp(exp_int_sum, min=1, max=2.0**self.m - 1)

        factor = floor_ste.apply((2.0**self.m - 1) / exp_int_sum)

        sigmoid_int = floor_ste.apply(exp_int * factor / (2.0 ** (self.m - self.output_bit + 1)))

        y_int = pre_x_int * sigmoid_int

        act_scaling_factor = act_scaling_factor * self.sigmoid_scaling_factor.to(x.device)
        self.act_scaling_factor = act_scaling_factor

        out = y_int * act_scaling_factor

        return out, act_scaling_factor

@track_init_args
class ISignGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8, n=23, lambda_scaler=6, temp = 1.702, channel_len=None):
        super(ISignGELU, self).__init__()
        self.output_bit = output_bit

        self.n = n  # sufficiently large integer
        self.lambda_scaler = lambda_scaler
        self.m = 31
        # The minimum value for ensuring accuracy (varies depending on models)

        scale_shape = channel_len if channel_len is not None else 1
        self.register_buffer("act_scaling_factor", nn.Parameter(torch.zeros(scale_shape)))
        self.register_buffer(
            "sigmoid_scaling_factor",
            nn.Parameter(torch.Tensor([1 / 2.0 ** (self.output_bit - 1)])),
        )
        self.temp = temp

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        scaled_floor = floor_ste.apply(x_int / 2)
        scaled_exp_floor = floor_ste.apply(x_int / (2.0**4))
        x_int = x_int + scaled_floor - scaled_exp_floor

        with torch.no_grad():
            x0_int = torch.ceil(1.0 / scaling_factor)

        min_x_int = torch.clamp(self.n * self.lambda_scaler * (-x0_int), min=-(2.0**self.m))
        x_int = torch.maximum(x_int, min_x_int)

        q = floor_ste.apply(x_int / (-x0_int))
        r = -(x_int - q * (-x0_int))

        exp_int = floor_ste.apply((-r) / 2)
        exp_int += x0_int
        exp_int = floor_ste.apply(exp_int * 2.0 ** (self.n - q))

        scaling_factor = scaling_factor / 2.0**self.n

        return exp_int, scaling_factor

    def forward(self, x, act_scaling_factor=None):
        pre_x_int = round_ste.apply(x / act_scaling_factor)
        scaling_factor_sig = act_scaling_factor * self.temp

        A = 2.0
        A_int = round_ste.apply(
            torch.tensor(A, device=x.device, dtype=x.dtype) / act_scaling_factor
        )

        pos_mask = pre_x_int > 0

        x_int_max_base = torch.clamp(pre_x_int, min=0)

        x_int_max = torch.where(
            pos_mask,
            pre_x_int + A_int,
            x_int_max_base
        )

        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)

        exp_int_sum = exp_int + exp_int_max
        exp_int_sum = torch.clamp(exp_int_sum, min=1, max=2.0**self.m - 1)

        factor = floor_ste.apply((2.0**self.m - 1) / exp_int_sum)

        sigmoid_int = floor_ste.apply(exp_int * factor / (2.0 ** (self.m - self.output_bit + 1)))

        x_int_out = pre_x_int * sigmoid_int

        out_scaling_factor = act_scaling_factor * self.sigmoid_scaling_factor
        self.act_scaling_factor = out_scaling_factor

        out = x_int_out * out_scaling_factor

        return out, out_scaling_factor

@track_init_args
class ISignGELUNoLambda(nn.Module):
    """
    ISignGELU variant without lambda scaling in the integer exp clamp.
    """

    def __init__(self, output_bit=8, n=23, temp=1.702, channel_len=None):
        super(ISignGELUNoLambda, self).__init__()
        self.output_bit = output_bit

        self.n = n  # sufficiently large integer
        self.m = 31

        scale_shape = channel_len if channel_len is not None else 1
        self.register_buffer("act_scaling_factor", nn.Parameter(torch.zeros(scale_shape)))
        self.register_buffer(
            "sigmoid_scaling_factor",
            nn.Parameter(torch.Tensor([1 / 2.0 ** (self.output_bit - 1)])),
        )
        self.temp = temp

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        scaled_floor = floor_ste.apply(x_int / 2)
        scaled_exp_floor = floor_ste.apply(x_int / (2.0**4))
        x_int = x_int + scaled_floor - scaled_exp_floor

        with torch.no_grad():
            x0_int = torch.ceil(1.0 / scaling_factor)

        min_x_int = torch.clamp(self.n * (-x0_int), min=-(2.0**self.m))
        x_int = torch.maximum(x_int, min_x_int)

        q = floor_ste.apply(x_int / (-x0_int))
        r = -(x_int - q * (-x0_int))

        exp_int = floor_ste.apply((-r) / 2)
        exp_int += x0_int
        exp_int = floor_ste.apply(exp_int * 2.0 ** (self.n - q))

        scaling_factor = scaling_factor / 2.0**self.n

        return exp_int, scaling_factor

    def forward(self, x, act_scaling_factor=None):
        pre_x_int = round_ste.apply(x / act_scaling_factor)
        scaling_factor_sig = act_scaling_factor * self.temp

        A = 2.0
        A_int = round_ste.apply(
            torch.tensor(A, device=x.device, dtype=x.dtype) / act_scaling_factor
        )

        pos_mask = pre_x_int > 0

        x_int_max_base = torch.clamp(pre_x_int, min=0)

        x_int_max = torch.where(
            pos_mask,
            pre_x_int + A_int,
            x_int_max_base
        )

        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)

        exp_int_sum = exp_int + exp_int_max
        exp_int_sum = torch.clamp(exp_int_sum, min=1, max=2.0**self.m - 1)

        factor = floor_ste.apply((2.0**self.m - 1) / exp_int_sum)

        sigmoid_int = floor_ste.apply(exp_int * factor / (2.0 ** (self.m - self.output_bit + 1)))

        x_int_out = pre_x_int * sigmoid_int

        out_scaling_factor = act_scaling_factor * self.sigmoid_scaling_factor
        self.act_scaling_factor = out_scaling_factor

        out = x_int_out * out_scaling_factor

        return out, out_scaling_factor

@track_init_args
class ISegGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8, n=23, lambda_scaler=6, temp = 1.702, channel_len=None):
        super(ISegGELU, self).__init__()
        self.output_bit = output_bit

        self.n = n  # sufficiently large integer
        self.lambda_scaler = lambda_scaler
        self.m = 31
        # The minimum value for ensuring accuracy (varies depending on models)

        scale_shape = channel_len if channel_len is not None else 1
        self.register_buffer("act_scaling_factor", nn.Parameter(torch.zeros(scale_shape)))
        self.register_buffer(
            "sigmoid_scaling_factor",
            nn.Parameter(torch.Tensor([1 / 2.0 ** (self.output_bit - 1)])),
        )
        self.temp = temp

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        scaled_floor = floor_ste.apply(x_int / 2)
        scaled_exp_floor = floor_ste.apply(x_int / (2.0**4))
        x_int = x_int + scaled_floor - scaled_exp_floor

        with torch.no_grad():
            x0_int = torch.ceil(1.0 / scaling_factor)

        min_x_int = torch.clamp(self.n * self.lambda_scaler * (-x0_int), min=-(2.0**self.m))
        x_int = torch.maximum(x_int, min_x_int)

        q = floor_ste.apply(x_int / (-x0_int))
        r = -(x_int - q * (-x0_int))

        exp_int = floor_ste.apply((-r) / 2)
        exp_int += x0_int
        exp_int = floor_ste.apply(exp_int * 2.0 ** (self.n - q))

        scaling_factor = scaling_factor / 2.0**self.n

        return exp_int, scaling_factor

    def forward(self, x, act_scaling_factor=None):
        pre_x_int = round_ste.apply(x / act_scaling_factor)
        scaling_factor_sig = act_scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)

        exp_int_sum = exp_int + exp_int_max

        # Clamp to avoid overflow or divide-by-zero
        exp_int_sum = torch.clamp(exp_int_sum, min=1, max=2.0**self.m - 1)

        factor = floor_ste.apply((2.0**self.m - 1) / exp_int_sum)

        sigmoid_int = floor_ste.apply(exp_int * factor / (2.0 ** (self.m - self.output_bit + 1)))

        x_int = pre_x_int * sigmoid_int

        act_scaling_factor = act_scaling_factor * self.sigmoid_scaling_factor
        self.act_scaling_factor = act_scaling_factor

        out = x_int * act_scaling_factor

        return out, act_scaling_factor

@track_init_args
class IViTSoftmax(nn.Module):
    """
    Implementation of Shiftmax
    Class to quantize given Softmax layer
    """

    def __init__(self, output_bit=8, **kwargs):
        super(IViTSoftmax, self).__init__()
        self.output_bit = output_bit

        self.n = 15  # sufficiently large integer
        self.m = 31
        # The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer(
            "act_scaling_factor",
            nn.Parameter(torch.Tensor([1 / 2.0 ** (self.output_bit - 1)])),
        )

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        scaled_floor = floor_ste.apply(x_int / 2)
        scaled_exp_floor = floor_ste.apply(x_int / 16)
        x_int = x_int + scaled_floor - scaled_exp_floor

        with torch.no_grad():
            x0_int = torch.ceil(1.0 / scaling_factor)

        min_x_int = torch.clamp(self.n * (-x0_int), min=-(2.0**self.m))
        x_int = torch.maximum(x_int, min_x_int)

        q = floor_ste.apply(x_int / (-x0_int))
        r = -(x_int - q * (-x0_int))

        exp_int = floor_ste.apply((-r) / 2)
        exp_int += x0_int
        exp_int = exp_int * 2.0 ** (self.n - q)
        # exp_int = floor_ste.apply(exp_int)
        exp_int = torch.clamp(exp_int, min=0)

        return exp_int

    def forward(self, x, act_scaling_factor, dynamic_input_scale=False):
        x_int = round_ste.apply(x / act_scaling_factor)
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int = self.int_exp_shift(x_int, act_scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2**self.m - 1)
        factor = floor_ste.apply((2.0**self.m - 1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / (2.0 ** (self.m - self.output_bit + 1)))

        return exp_int * self.act_scaling_factor, self.act_scaling_factor


@track_init_args
class IViTSoftmaxV2(nn.Module):
    """
    Shiftmax variant that caps M for 32-bit hardware and right-shifts Iexp.

    Calibration represents a large normalization range as (M, R). M is capped
    at hardware_m_max, while Iexp is shifted right by R before its denominator
    sum is accumulated.
    """

    def __init__(
        self,
        output_bit=8,
        n=15,
        lambda_scaler=1,
        input_bit=8,
        calibrate_m=True,
        m_guard_bits=4,
        m_calib_mode="ema_percentile",
        m_percentile=99.9,
        m_ema_momentum=0.95,
        m_max=63,
        hardware_m_max=31,
        **kwargs,
    ):
        super(IViTSoftmaxV2, self).__init__()
        self.output_bit = output_bit
        self.input_bit = input_bit
        self.n = n
        self.lambda_scaler = lambda_scaler
        self.calibrate_m = calibrate_m
        self.m_guard_bits = int(m_guard_bits)
        self.m_calib_mode = str(m_calib_mode)
        self.m_percentile = float(m_percentile)
        self.m_ema_momentum = float(m_ema_momentum)
        self.m_max = int(m_max)
        self.running_stat = True
        self.hardware_m_max = int(hardware_m_max)
        self.register_buffer(
            "act_scaling_factor",
            nn.Parameter(torch.Tensor([1 / 2.0 ** (self.output_bit - 1)])),
        )
        # V2 fixes denominator M to the hardware-safe value. Calibration only
        # observes the extra logical range as R and exp_clamp_m.
        self.m = self.hardware_m_max
        self.r = 0
        self.register_buffer("max_exp_int_sum", torch.tensor(0.0))
        self.register_buffer("percentile_exp_int_sum", torch.tensor(0.0))
        self.register_buffer("ema_exp_int_sum", torch.tensor(0.0))
        self.register_buffer("calibrated_m", torch.tensor(self.hardware_m_max, dtype=torch.long))
        self.register_buffer("m_calib_batches", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibrated_r", torch.tensor(0, dtype=torch.long))
        self.exp_clamp_m = self.hardware_m_max
        self.register_buffer(
            "calibrated_exp_clamp_m",
            torch.tensor(self.exp_clamp_m, dtype=torch.long),
        )

    def _set_fixed_hardware_m(self):
        self.m = self.hardware_m_max
        self.calibrated_m.fill_(self.hardware_m_max)

    def fix(self):
        self._set_fixed_hardware_m()
        if self.calibrate_m and self.max_exp_int_sum.item() > 0:
            self.r = int(self.calibrated_r.item())
            self.exp_clamp_m = int(self.calibrated_exp_clamp_m.item())
        self.running_stat = False

    def unfix(self):
        self.running_stat = True

    def reset_m_calibration(self):
        self.max_exp_int_sum.zero_()
        self.percentile_exp_int_sum.zero_()
        self.ema_exp_int_sum.zero_()
        self.m_calib_batches.zero_()
        self._set_fixed_hardware_m()
        self.calibrated_r.zero_()
        self.calibrated_exp_clamp_m.fill_(self.hardware_m_max)
        self.r = 0
        self.exp_clamp_m = self.hardware_m_max

    def _sync_m_from_calibration(self):
        self._set_fixed_hardware_m()
        if self.calibrate_m and self.max_exp_int_sum.item() > 0:
            self.r = int(self.calibrated_r.item())
            self.exp_clamp_m = int(self.calibrated_exp_clamp_m.item())
        return self.m, self.r, self.exp_clamp_m

    def _update_m_calibration(self, exp_int_sum):
        if not self.calibrate_m or not self.running_stat:
            return self.m, self.r

        with torch.no_grad():
            exp_values = exp_int_sum.detach().float().reshape(-1)
            observed_max = exp_values.amax()
            self.max_exp_int_sum.copy_(torch.maximum(self.max_exp_int_sum, observed_max))

            if self.m_calib_mode == "percentile":
                q = min(max(self.m_percentile / 100.0, 0.0), 1.0)
                observed_ref = torch.quantile(exp_values, q)
                self.percentile_exp_int_sum.copy_(
                    torch.maximum(self.percentile_exp_int_sum, observed_ref)
                )
                calib_ref = self.percentile_exp_int_sum
            elif self.m_calib_mode == "ema_percentile":
                q = min(max(self.m_percentile / 100.0, 0.0), 1.0)
                observed_ref = torch.quantile(exp_values, q)
                self.percentile_exp_int_sum.copy_(
                    torch.maximum(self.percentile_exp_int_sum, observed_ref)
                )
                if self.m_calib_batches.item() == 0:
                    self.ema_exp_int_sum.copy_(observed_ref)
                else:
                    self.ema_exp_int_sum.copy_(
                        self.ema_exp_int_sum * self.m_ema_momentum
                        + observed_ref * (1.0 - self.m_ema_momentum)
                    )
                calib_ref = self.ema_exp_int_sum
            elif self.m_calib_mode == "max":
                calib_ref = self.max_exp_int_sum
            else:
                raise ValueError(f"unknown softmax m calibration mode: {self.m_calib_mode}")

            sum_bits = torch.ceil(torch.log2(calib_ref.clamp_min(1.0))).long()
            required_m = torch.clamp(
                sum_bits + self.m_guard_bits,
                min=self.hardware_m_max,
                max=self.m_max,
            )
            # M is fixed to the hardware-safe denominator width. R carries only
            # the extra logical range needed beyond that fixed M.
            calibrated_r = torch.clamp(
                required_m - self.hardware_m_max,
                min=0,
            )
            self.calibrated_m.fill_(self.hardware_m_max)
            self.calibrated_r.copy_(calibrated_r)
            self.calibrated_exp_clamp_m.copy_(required_m)
            self.m_calib_batches += 1

        return int(self.calibrated_m.item()), int(self.calibrated_r.item())

    def int_exp_shift(self, x_int, scaling_factor):
        scaled_floor = floor_ste.apply(x_int / 2)
        scaled_exp_floor = floor_ste.apply(x_int / 16)
        x_int = x_int + scaled_floor - scaled_exp_floor

        with torch.no_grad():
            x0_int = torch.ceil(1.0 / scaling_factor)

        # During calibration, use the observed logical/high-M clamp to estimate
        # the V1-equivalent exponent range and derive R. During fixed inference,
        # use the hardware clamp: the integer exp path must not require values
        # below -2^31 on 32-bit hardware.
        clamp_m = (
            self.exp_clamp_m
            if self.calibrate_m and self.running_stat
            else self.hardware_m_max
        )
        min_x_int = torch.clamp(
            self.n * self.lambda_scaler * (-x0_int),
            min=-(2.0**clamp_m),
        )
        x_int = torch.maximum(x_int, min_x_int)

        q = floor_ste.apply(x_int / (-x0_int))
        r = -(x_int - q * (-x0_int))

        exp_int = floor_ste.apply((-r) / 2)
        exp_int += x0_int
        exp_int = exp_int * 2.0 ** (self.n - q)
        exp_int = torch.clamp(exp_int, min=0)

        return exp_int

    def forward(self, x, act_scaling_factor, dynamic_input_scale=False):
        self._sync_m_from_calibration()

        if torch.is_tensor(act_scaling_factor) and act_scaling_factor.numel() > 1:
            scale = act_scaling_factor.reshape(-1)
            if scale.numel() == x.shape[-1]:
                scale = scale.reshape(*([1] * (x.dim() - 1)), x.shape[-1])
                x = round_ste.apply(x / scale) * scale
                act_scaling_factor = scale.max()
            elif x.dim() >= 2 and scale.numel() == x.shape[-2] * x.shape[-1]:
                scale = scale.reshape(*([1] * (x.dim() - 2)), x.shape[-2], x.shape[-1])
                x = round_ste.apply(x / scale) * scale
                act_scaling_factor = scale.max(dim=-1, keepdim=True).values
            else:
                act_scaling_factor = scale.max()

        x_int = round_ste.apply(x / act_scaling_factor)
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int = self.int_exp_shift(x_int, act_scaling_factor)
        if self.calibrate_m and self.running_stat:
            raw_exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
            m, r = self._update_m_calibration(raw_exp_int_sum)
        else:
            m, r = self.m, self.r

        if r > 0:
            divisor = 2.0 ** r
            # Float64 preserves the integer low bits used by the remainder.
            # The fake-quant tensors are otherwise float32, which cannot
            # represent every integer once Iexp exceeds 2^24.
            exp_int_for_sum = exp_int.to(torch.float64)
            exp_int_quotient = floor_ste.apply(exp_int_for_sum / divisor)
            exp_int_remainder = (
                exp_int_for_sum - exp_int_quotient * divisor
            )
            exp_int_sum = exp_int_quotient.sum(dim=-1, keepdim=True)
            exp_int_sum = exp_int_sum + floor_ste.apply(
                exp_int_remainder.sum(dim=-1, keepdim=True) / divisor
            )
        else:
            exp_int_sum = exp_int.to(torch.float64).sum(dim=-1, keepdim=True)

        exp_int_sum = exp_int_sum.clamp(min=1, max=2.0**m - 1)
        factor = floor_ste.apply((2.0**m - 1) / exp_int_sum.clamp_min(1))
        factor = factor.to(exp_int.dtype)
        exp_int = floor_ste.apply(
            exp_int
            * factor
            / (2.0 ** (m - self.output_bit + 1 + r))
        )

        return exp_int * self.act_scaling_factor, self.act_scaling_factor

class IViTLayerScale(nn.Module):
    def __init__(
        self,
        config,
        weight_bit=8,
        quant_mode="symmetric",
    ) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(
            config.layerscale_value * torch.ones(config.hidden_size)
        )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        # --- buffers, same pattern as IViTLinear ---
        self.register_buffer("scaling_factor", torch.ones(config.hidden_size))
        self.register_buffer("lambda1_integer", torch.ones_like(self.lambda1))

    def forward(self, hidden_state: torch.Tensor, act_scaling_factor=None):

        # Step 1 — compute scale for lambda1 (per-channel, same as IViTLinear)
        with torch.no_grad():
            cur_min = self.lambda1.min()   # scalar — lambda1 is 1D so per-tensor is fine
            cur_max = self.lambda1.max()

            self.scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit,
                cur_min.unsqueeze(0),
                cur_max.unsqueeze(0),
            )

        # Step 2 — quantize lambda1 to integer
        self.lambda1_integer = self.weight_function(
            self.lambda1, self.weight_bit, self.scaling_factor, True
        )

        # Step 3 — quantize incoming hidden_state using act_scaling_factor
        hidden_state_int = round_ste.apply(hidden_state / act_scaling_factor)

        # Step 4 — integer multiply
        # (batch, seq_len, hidden_size) * (hidden_size,)
        out = hidden_state_int * self.lambda1_integer
        out = round_ste.apply(out)

        # Step 5 — rescale back to float
        out_scaling_factor = self.scaling_factor * act_scaling_factor
        out = out * out_scaling_factor

        return out, out_scaling_factor
    

def quant_ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """ "for debug and test only, need to use cuda version instead"""
    # B, n_heads, head_dim, N
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(value_spatial_shapes):
        # B, n_heads, head_dim, H, W
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        # B, Len_q, n_heads, P, 2 -> B, n_heads, Len_q, P, 2 -> B*n_heads, Len_q, P, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # B*n_heads, head_dim, Len_q, P
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (B, Len_q, n_heads, L * P) -> (B, n_heads, Len_q, L, P) -> (B*n_heads, 1, Len_q, L*P)
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    # B*n_heads, head_dim, Len_q, L*P
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()
 

@dataclass
class QuantizedBaseModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor = None
    act_scaling_factor: torch.Tensor = None      # ← added
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None

@dataclass
class QuantizedBaseModelOutputWithPooling(ModelOutput):
    act_scaling_factor: torch.Tensor = None      # ← added
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@track_init_args
class IViTExp(nn.Module):
    """
    ShiftExp variant with separate negative and positive integer paths.

    Negative inputs keep the original I-ViT ShiftExp approximation.
    Positive inputs use a separate positive decomposition, which is needed
    when exp() receives positive values, for example in box refinement.

    The function returns one common output scale. Here, both branches are
    represented using the positive-branch natural scale S.
    """

    def __init__(self, output_bit=None, n=23, clamp_min_real=-20.0, clamp_max_real=20.0):
        super().__init__()
        self.output_bit = output_bit
        self.n = n
        self.clamp_min_real = clamp_min_real
        self.clamp_max_real = clamp_max_real

        self.register_buffer("act_scaling_factor", torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def _log2_input_shift(self, x_int):
        """
        Approximate multiplication by log2(e).

        e^x = 2^(x log2(e))

        log2(e) ≈ 1 + 2^-1 - 2^-4 + 2^-8 + 2^-9
        """
        return (
            x_int
            + floor_ste.apply(x_int / 2)
            - floor_ste.apply(x_int / 2**4)
            + floor_ste.apply(x_int / 2**8)
            + floor_ste.apply(x_int / 2**9)
        )

    def _int_exp_shift_negative(self, x_int, scaling_factor):
        """
        Negative branch.

        Assumes x_int <= 0 after log2(e) approximation.

        Natural output scale:
            S_exp_neg = S / 2^n

        Returned integer:
            exp_int_neg
        """
        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)

        # Avoid too negative values, which would be rounded to zero anyway.
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q

        # Residual approximation on [-1, 0]:
        # 2^u ≈ 1 + u / 2
        exp_int = r / 2 - x0_int

        # Apply 2^(n-q), then floor to integer.
        exp_int = torch.clamp(
            floor_ste.apply(exp_int * 2.0 ** (self.n - q)),
            min=0
        )

        return exp_int

    def _int_exp_shift_positive(self, x_int, scaling_factor):
        """
        Positive branch.

        Assumes x_int >= 0 after log2(e) approximation.

        Uses:
            q = floor(x_int / x0_int)
            r = x_int - q x0_int

        with:
            x0_int ≈ 1 / S

        Then:
            2^(S x_int) ≈ 2^q (1 + S r)
                          ≈ S (x0_int + r) 2^q

        Natural output scale:
            S_exp_pos = S

        Returned integer:
            exp_int_pos = (x0_int + r) 2^q
        """
        with torch.no_grad():
            # Match the magnitude used by the original negative-sign
            # formulation: floor(-1 / S) == -ceil(1 / S).
            x0_int = torch.ceil(1.0 / scaling_factor)
            x0_int = torch.clamp(x0_int, min=1)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q

        # Residual approximation on [0, 1):
        # 2^u ≈ 1 + u
        exp_int = x0_int + r

        # Apply 2^q as a left shift.
        exp_int = torch.clamp(
            floor_ste.apply(exp_int * 2.0 ** q),
            min=0
        )

        return exp_int

    def int_exp_shift(self, x_int, scaling_factor):
        """
        Integer exponential approximation.

        The sign must be checked after the log2(e) approximation, because
        the decomposition is applied to I_p, not to the original integer input.
        """

        # Approximate I * log2(e)
        x_int = self._log2_input_shift(x_int)

        # Branch according to I_p
        pos_mask = x_int >= 0

        exp_int = torch.empty_like(x_int)

        if torch.any(pos_mask):
            exp_int[pos_mask] = self._int_exp_shift_positive(
                x_int[pos_mask],
                scaling_factor
            )

        if torch.any(~pos_mask):
            neg_exp_int = self._int_exp_shift_negative(
                x_int[~pos_mask],
                scaling_factor
            )

            # Negative branch natural scale is S / 2^n.
            # To return one common scale S, convert:
            #
            #   neg_exp_int * (S / 2^n)
            # = (neg_exp_int / 2^n) * S
            #
            # This keeps the same real value, but the returned exp_int is
            # not strictly integer-like for negative values.
            exp_int[~pos_mask] = neg_exp_int / 2.0**self.n

        # Common output scale.
        exp_scale = scaling_factor

        return exp_int, exp_scale

    def forward(self, x, act_scaling_factor=None):
        assert act_scaling_factor is not None, "act_scaling_factor must be provided"

        # Convert real-valued tensor to integer representation.
        x_int = round_ste.apply(x / act_scaling_factor)

        # Optional real-domain clamp.
        if self.clamp_min_real is not None or self.clamp_max_real is not None:
            with torch.no_grad():
                lo = None
                hi = None

                if self.clamp_min_real is not None:
                    lo = torch.floor(
                        torch.tensor(
                            self.clamp_min_real,
                            device=x.device,
                            dtype=torch.float32
                        ) / act_scaling_factor
                    )

                if self.clamp_max_real is not None:
                    hi = torch.ceil(
                        torch.tensor(
                            self.clamp_max_real,
                            device=x.device,
                            dtype=torch.float32
                        ) / act_scaling_factor
                    )

            if lo is not None and hi is not None:
                x_int = torch.clamp(x_int, lo.item(), hi.item())
            elif lo is not None:
                x_int = torch.max(x_int, lo.item())
            elif hi is not None:
                x_int = torch.min(x_int, hi.item())

        # Integer exp approximation.
        exp_int, exp_scale = self.int_exp_shift(x_int, act_scaling_factor)

        # Optional output clipping.
        if self.output_bit is not None:
            max_int = 2 ** (self.output_bit - 1) - 1
            exp_int = torch.clamp(exp_int, 0, max_int)

        self.act_scaling_factor = exp_scale

        return exp_int * exp_scale, exp_scale
