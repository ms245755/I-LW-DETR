from torch.autograd import Function
import torch


def linear_quantize(input, scale, zero_point, is_weight):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    Parameters:
    ----------
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if is_weight:
        if len(input.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
            # reshape scale and zeropoint for linear weights
        elif len(input.shape) == 2:
            scale = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        else:
            scale = scale.view(-1)
            zero_point = zero_point.view(-1)
    else:
        if len(input.shape) == 2:
            scale = scale.view(1, -1)
            zero_point = zero_point.view(1, -1)
        elif len(input.shape) == 3:
            scale = scale.view(1, 1, -1)
            zero_point = zero_point.view(1, 1, -1)
        elif len(input.shape) == 4:
            scale = scale.view(1, -1, 1, 1)
            zero_point = zero_point.view(1, -1, 1, 1)
        elif len(input.shape) == 1:
            pass
        else:
            raise NotImplementedError

    # quantized = float / scale + zero_point
    return round_ste.apply(1.0 / scale * input + zero_point)


def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.
    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    # in this part, we do not need any gradient computation,
    # in order to enfore this, we put torch.no_grad()
    with torch.no_grad():
        n = 2.0 ** (num_bits - 1) - 1
        eps = torch.finfo(torch.float32).eps

        max_val = torch.max(-min_val, max_val)
        scale = max_val / float(n)
        scale.clamp_(eps)

    return scale


class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """

        scale = specified_scale

        zero_point = torch.tensor(0.0).to(x.device)

        n = 2.0 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, is_weight=is_weight)
        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
        ctx.is_weight = is_weight
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        is_weight = ctx.is_weight
        if is_weight:
            if len(grad_output.shape) == 4:
                scale = scale.view(-1, 1, 1, 1)
            elif len(grad_output.shape) == 2:
                scale = scale.view(-1, 1)
            else:
                scale = scale.view(-1)
        else:
            if len(grad_output.shape) == 2:
                scale = scale.view(1, -1)
            elif len(grad_output.shape) == 3:
                scale = scale.view(1, 1, -1)
            elif len(grad_output.shape) == 4:
                scale = scale.view(1, -1, 1, 1)
            elif len(grad_output.shape) == 1:
                pass
            else:
                raise NotImplementedError
        return grad_output / scale, None, None, None


class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.
    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)

    # Get mantissa and exponent with torch.frexp
    mantissa, exponent = torch.frexp(inputs)

    # Scale mantissa by 2^max_bit and round it
    max_bit_factor = 2.0**max_bit
    mantissa *= max_bit_factor
    mantissa = mantissa.round()
    mantissa = mantissa.double()
    mantissa = mantissa.view(shape_of_input)

    # Adjust exponent as needed
    exponent = max_bit - exponent
    exponent = exponent.long()
    exponent = exponent.view(shape_of_input)

    return mantissa, exponent


class fixedpoint_mul(Function):
    """
    Function to perform fixed-point arthmetic that can match integer arthmetic on hardware.
    Parameters:
    ----------
    pre_act: input tensor
    pre_act_scaling_factor: ithe scaling factor of the input tensor
    bit_num: quantization bitwidth
    quant_mode: The mode for quantization, 'symmetric' or 'asymmetric'
    z_scaling_factor: the scaling factor of the output tensor
    identity: identity tensor
    identity_scaling_factor: the scaling factor of the identity tensor
    """

    @staticmethod
    def forward(
        ctx,
        pre_act,
        pre_act_scaling_factor,
        bit_num,
        quant_mode,
        z_scaling_factor,
        identity=None,
        identity_scaling_factor=None,
    ):

        # TODO(Sehoon): May require other type of reshape
        if len(pre_act.shape) == 2:
            reshape = lambda x: x.view(1, -1)
        elif len(pre_act.shape) == 3:
            reshape = lambda x: x.view(1, 1, -1)
        elif len(pre_act.shape) == 4:
            reshape = lambda x: x.view(1, -1, 1, 1)
        elif len(pre_act.shape) == 1:
            pass
        else:
            raise NotImplementedError
        ctx.identity = identity

        n = 2.0 ** (bit_num - 1) - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            ctx.z_scaling_factor = z_scaling_factor

            # Quantize pre-activation
            z_int = pre_act / pre_act_scaling_factor
            z_int = z_int.round()
            z_int = z_int.double()

            # New scale
            _A = pre_act_scaling_factor.double()
            _B = z_scaling_factor.double()
            new_scale = _A / _B
            new_scale = reshape(new_scale)

            m, e = batch_frexp(new_scale)
            z_int *= m
            z_int /= 2.0**e
            z_int = z_int.round()
            z_int = z_int.float()

            if bit_num in [4, 8, 16, 32]:
                if quant_mode == "symmetric":
                    return torch.clamp(z_int, -n - 1, n)
                else:
                    return torch.clamp(z_int, 0, n)
            else:
                return z_int

    @staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        z_scaling_factor = ctx.z_scaling_factor

        # In-place operation to reduce memory usage
        grad_output = grad_output / z_scaling_factor  # In-place division

        if ctx.identity is not None:
            identity_grad = grad_output  # Don't modify grad_output in-place for identity

        return grad_output, None, None, None, None, identity_grad, None

def _stackable_scale(scale):
    if not torch.is_tensor(scale):
        return torch.tensor(scale)
    if scale.numel() == 1:
        return scale.reshape(())
    return scale


def _max_compatible_scale(scales):
    scales = [_stackable_scale(scale) for scale in scales]
    return torch.stack([scale.max() for scale in scales]).max()


def concat_list_tensor_scale(tensor_list, dim = 0):
    max_scale = _max_compatible_scale([scale for (_, scale) in tensor_list])

    result = []
    for tmp, scale in tensor_list:
        result.append(round_ste.apply(tmp / max_scale))
    
    result = torch.cat(result, dim = dim)
    result = result * max_scale
    return result, max_scale

def add_tensor_scale(x, scale_x, y, scale_y):
    max_scale = _max_compatible_scale([scale_x, scale_y])
    x = round_ste.apply(x / max_scale)
    y = round_ste.apply(y / max_scale)

    result = x + y
    return result * max_scale, max_scale

def multiply_tensor_scale(x, scale_x, y, scale_y):
    max_scale = scale_x * scale_y
    result = x * y
    return result, max_scale
