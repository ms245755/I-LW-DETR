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

        if self.bias is not None:
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
    """

    def __init__(
        self,
        activation_bit=8,
        act_range_momentum=0.99,
        running_stat=True,
        per_channel=False,
        quant_mode="symmetric",
    ):
        super(IViTAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.per_channel = per_channel

        self.register_buffer("act_scaling_factor", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("out_scaling_factor", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("min_val", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("max_val", nn.Parameter(torch.tensor(1.0)))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return (
            "{0}(activation_bit={1}, "
            "quant_mode: {2}, Act_min: {3:.2f}, "
            "Act_max: {4:.2f})".format(
                self.__class__.__name__,
                self.activation_bit,
                self.quant_mode,
                self.min_val.item(),
                self.max_val.item(),
            )
        )

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

    def forward(self, x, act_scaling_factor=None, x_2=None, act_scaling_factor_2=None):
        # collect runnng stats
        with torch.no_grad():
            x_act = x if x_2 is None else x_2 + x
            if self.running_stat:
                if len(x_act.shape) == 4:
                    x_act = x_act.permute(0, 2, 3, 1)
                v = x_act.reshape(-1, x_act.shape[-1])
                v = v.transpose(0, 1)

                cur_min = v.min(dim=1).values
                cur_max = v.max(dim=1).values
                if torch.eq(self.min_val, self.max_val).all():
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    self.min_val = self.min_val * self.act_range_momentum + cur_min * (
                        1 - self.act_range_momentum
                    )
                    self.max_val = self.max_val * self.act_range_momentum + cur_max * (
                        1 - self.act_range_momentum
                    )
                self.max_val = self.max_val.max()
                self.min_val = self.min_val.min()

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val
            )

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

        return quant_act_int * correct_output_scale, correct_output_scale


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
class IViTLayerNorm(nn.LayerNorm):
    """
    Implementation of I-LayerNorm
    Class to quantize given LayerNorm layer
    """

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super(IViTLayerNorm, self).__init__(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )
        self.register_buffer("norm_scaling_factor", nn.Parameter(torch.ones(normalized_shape)))
        self.register_buffer("bias_integer", nn.Parameter(torch.ones_like(self.bias)))
        self.register_buffer(
            "base_act_scaling_factor",
            nn.Parameter(torch.sqrt(torch.tensor(normalized_shape, dtype=torch.float)) / 2.0**30),
        )

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, act_scaling_factor=None):
        # Normalization: computes mean and variance(std)
        x_int = round_ste.apply(x / act_scaling_factor)
        mean_int = floor_ste.apply(x_int.sum(dim=2, keepdim=True) / x_int.shape[2])
        y_int = x_int - mean_int
        y_sq_int = y_int**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2.0**16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)
            k = k_1
        std_int = k

        exp = floor_ste.apply((2.0**31 - 1) / std_int)
        factor = exp
        y_int = floor_ste.apply(y_int * factor / 2)

        # scaling and shifting
        bias = self.bias / self.weight
        bias_int = floor_ste.apply(bias / self.base_act_scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int
        act_scaling_factor = self.base_act_scaling_factor * self.weight
        x = y_int * act_scaling_factor
        self.norm_scaling_factor = act_scaling_factor

        return x, act_scaling_factor
    
@track_init_args
class IViTLayerNormChannelFirst(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super(IViTLayerNormChannelFirst, self).__init__(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )
        self.register_buffer("norm_scaling_factor", nn.Parameter(torch.ones(normalized_shape)))
        self.register_buffer("bias_integer", nn.Parameter(torch.ones_like(self.bias)))
        self.register_buffer(
            "base_act_scaling_factor",
            nn.Parameter(torch.sqrt(torch.tensor(normalized_shape, dtype=torch.float)) / 2.0**30),
        )
    def fix(self):
        pass

    def unfix(self):
        pass
    
    def forward(self, x, act_scaling_factor=None):
        # Normalization: computes mean and variance(std)
        x_int = round_ste.apply(x / act_scaling_factor)
        mean_int = floor_ste.apply(x_int.sum(dim=1, keepdim=True) / x_int.shape[1])
        y_int = x_int - mean_int
        y_sq_int = y_int**2
        var_int = torch.sum(y_sq_int, axis=1, keepdim=True)

        # Integer Iteration
        k = 2.0**16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)
            k = k_1
        std_int = k

        exp = floor_ste.apply((2.0**31 - 1) / std_int)
        factor = exp
        y_int = floor_ste.apply(y_int * factor / 2)

        # scaling and shifting
        bias = self.bias / self.weight
        bias_int = floor_ste.apply(bias / self.base_act_scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int[:, None, None]
        act_scaling_factor = self.base_act_scaling_factor * self.weight[:, None, None]
        x = y_int * act_scaling_factor
        self.norm_scaling_factor = act_scaling_factor

        return x, act_scaling_factor
    
class IViTGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8):
        super(IViTGELU, self).__init__()
        self.output_bit = output_bit

        self.n = 23  # sufficiently large integer
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
class ISegGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8, n=23, lambda_scaler=6):
        super(ISegGELU, self).__init__()
        self.output_bit = output_bit

        self.n = n  # sufficiently large integer
        self.lambda_scaler = lambda_scaler
        self.m = 31
        # The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer("act_scaling_factor", nn.Parameter(torch.zeros(1)))
        self.register_buffer(
            "sigmoid_scaling_factor",
            nn.Parameter(torch.Tensor([1 / 2.0 ** (self.output_bit - 1)])),
        )

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

    def forward(self, x, act_scaling_factor):
        x_int = round_ste.apply(x / act_scaling_factor)
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int = self.int_exp_shift(x_int, act_scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        # exp_int_sum.clamp_max_(2**self.m - 1)
        factor = floor_ste.apply((2.0**self.m - 1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / (2.0 ** (self.m - self.output_bit + 1)))

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