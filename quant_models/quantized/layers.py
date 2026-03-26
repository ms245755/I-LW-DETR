import torch
import torch.nn as nn


def batch_frexp(inputs, max_bit=31):
    """
    Decompose the input scaling factor into mantissa and exponent
    using base-2 logarithm (without torch.frexp).

    Parameters:
    ----------
    inputs : torch.Tensor
        Positive floating-point scaling factors.
    max_bit : int
        Number of bits to scale mantissa.

    Returns:
    -------
    mantissa : torch.Tensor
        Integer mantissa scaled by 2^max_bit.
    exponent : torch.Tensor
        Integer exponent such that:
            inputs ≈ mantissa * 2^(-exponent) / 2^max_bit
    """
    eps = 1e-8  # Prevent log(0)
    shape = inputs.shape
    inputs = inputs.view(-1)

    # Compute exponent: floor(log2(abs(x)))
    log2 = torch.log2(inputs + eps)
    exponent = torch.floor(log2)
    exponent = exponent.to(dtype=torch.long)

    # Compute mantissa: x / (2^exponent)
    mantissa = inputs / (2.0**exponent)

    # Scale mantissa and round
    mantissa = torch.round(mantissa * (2**max_bit)).double()
    exponent = (max_bit - exponent).view(shape)
    mantissa = mantissa.view(shape)

    return mantissa, exponent


def linear_quantize(input, scale, zero_point, is_weight=False):
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
        else:
            raise NotImplementedError

    # quantized = float / scale + zero_point
    return torch.round(1.0 / scale * input + zero_point)


def symmetric_quantization_function(x, k, scale, zero_point):
    """
    Performs symmetric quantization on the input tensor `x`.

    Args:
        x (Tensor): The input tensor (float32).
        k (int): Bit-width to quantize to (e.g., 8 for int8).
        scale (Tensor or float): Scaling factor for quantization.
        zero_point (Tensor or float): Offset to center the range around zero.

    Returns:
        Tensor: Quantized int32 tensor.
    """

    # Compute the maximum representable positive integer value for bit-width `k`
    # Example: k=8 → n=127 (for int8 quantization range [-128, 127])
    n = (1 << (k - 1)) - 1

    # Perform quantization:
    # x / scale → scale input to integer range
    # + zero_point → align around zero-point
    # round() → convert to nearest integer
    # to(int32) → ensure consistent dtype
    quantized = linear_quantize(x, scale, zero_point).to(torch.int32)
    # quantized = torch.round(x / scale + zero_point).to(torch.int32)  # FLOAT OP

    # Clamp the values to valid range: [-n - 1, n]
    # This keeps the result within representable bounds of the quantized bit width
    quantized = torch.clamp(quantized, -n - 1, n)

    return quantized


def fixed_point_multiplication(pre_act, pre_act_scaling_factor, bit_num, z_scaling_factor):
    """
    Rescales an already-quantized tensor from one scale to another using fixed-point math.

    Args:
        pre_act (Tensor): The input tensor (usually int-represented float values).
        pre_act_scaling_factor (Tensor): Original scaling factor for pre_act.
        bit_num (int): Bit-width used for quantization.
        z_scaling_factor (Tensor): Target scaling factor for the output.

    Returns:
        Tensor: The rescaled tensor, still in float32 domain but as if it were fixed-point scaled.
    """

    # Reshape scaling factors to be broadcastable with input based on dimensionality
    if len(pre_act.shape) == 2:
        reshape = lambda x: x.view(1, -1)
    elif len(pre_act.shape) == 3:
        reshape = lambda x: x.view(1, 1, -1)
    elif len(pre_act.shape) == 4:
        reshape = lambda x: x.view(1, -1, 1, 1)
    else:
        raise NotImplementedError  # Add more reshape rules if needed for other dims

    # Max representable integer for given bit width
    n = 2 ** (bit_num - 1) - 1

    with torch.no_grad():
        # Broadcast the scaling factor to match input shape
        pre_act_scaling_factor = reshape(pre_act_scaling_factor)

        # Convert float input to "quantized" int64 by dividing by its scale
        z_int = (pre_act / pre_act_scaling_factor).to(torch.int64)

        # Compute the scale ratio: old_scale / new_scale
        new_scale = reshape((pre_act_scaling_factor / z_scaling_factor))

        # Decompose the scale ratio into mantissa and exponent
        # such that new_scale ≈ m * 2^e
        m, e = batch_frexp(new_scale)

        # Apply fixed-point multiplication with m
        z_int = z_int * m

        # Right-shift by e (equivalent to division by 2^e) for fixed-point adjust
        z_int = (z_int / (2.0**e)).to(torch.float32)  # FLOAT OP

        # Clamp to representable range for given bit width
        return torch.clamp(z_int, -n - 1, n)


class QuantAct(nn.Module):
    """
    Module for quantizing activations during inference.
    Handles both symmetric quantization of raw activations and
    rescaling of already-quantized activations.
    """

    def __init__(
        self,
        activation_bit=8,
    ):
        super().__init__()

        self.activation_bit = activation_bit

        # Buffers to hold scaling and zero point values for quantization
        self.register_buffer("act_scaling_factor", None)  # Input scale
        self.register_buffer("zero_point", torch.tensor(0.0))
        # self.register_buffer("out_scaling_factor", None)  # Output scale (for quantize)
        # self.register_buffer("zero_point", None)  # Zero point for symmetric quantization

    def set_quant_params(self, act_scaling_factor, out_scaling_factor=None):
        """
        Set fixed quantization parameters for inference.
        """
        self.act_scaling_factor = act_scaling_factor
        # self.out_scaling_factor = out_scaling_factor
        # if self.out_scaling_factor is not None:
        #     self.zero_point = torch.zeros_like(
        #         self.out_scaling_factor
        #     )  # For symmetric quantization

    def __repr__(self):
        return f"{self.__class__.__name__}(activation_bit={self.activation_bit})"

    def forward(self, x, act_scaling_factor=None):
        """
        Forward pass to quantize and rescale activations.
        Args:
            x: Input tensor (float)
            act_scaling_factor: If provided, rescale from act_scaling_factor to self.act_scaling_factor
        Returns:
            Tuple of:
              - Dequantized float tensor (after applying quantization and scaling)
              - Scaling factor used (self.act_scaling_factor)
        """
        if act_scaling_factor is None:
            # Quantize float input x to integer domain (output scaling assumed)
            quant_act_int = symmetric_quantization_function(
                x, self.activation_bit, self.act_scaling_factor, self.zero_point
            )
        else:
            # Adjust scaling factor of already quantized input
            quant_act_int = fixed_point_multiplication(
                x, act_scaling_factor, self.activation_bit, self.act_scaling_factor
            )

        # Final dequantization (or scale propagation)
        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor


class QuantConv2d(nn.Module):
    """
    A frozen Conv2d module for quantized inference.
    Accepts integer weights/bias and performs fake quantization on activations.
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
        quant_mode="symmetric",
    ):
        super().__init__()

        # Standard Conv2d layer (weights/bias will be frozen)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Disable gradient updates (weights/bias are fixed)
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

        # Store conv scaling factor as buffer (non-trainable tensor)
        self.register_buffer("conv_scaling_factor", None)

    def set_quant_params(self, weight_int, bias_int, conv_scale):
        """
        Set quantized weights/bias and the conv scaling factor.
        Assumes inputs are integer tensors representing quantized params.
        """
        with torch.no_grad():
            self.conv.weight.copy_(weight_int.to(torch.float32))
            self.conv.bias.copy_(bias_int.to(torch.float32))
        self.conv_scaling_factor = conv_scale

    def forward(self, x, act_scaling_factor=None):
        """
        Forward pass with fake quantization.
        - x is dequantized using act_scaling_factor
        - Output is requantized using combined scale
        """
        act_scaling_factor = act_scaling_factor.view(1, -1, 1, 1)  # Match input shape
        bias_scaling_factor = self.conv_scaling_factor * act_scaling_factor
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        # Dequantize input
        x_int = torch.round(x / act_scaling_factor)  # Simulate quantized input

        # Perform convolution with frozen float weights
        x_out = self.conv(x_int)

        # Requantize output
        x_out = x_out * correct_output_scale  # Simulate quantized output

        return x_out, correct_output_scale


class QuantLinear(nn.Module):
    """
    A frozen Linear layer for quantized inference.
    Takes quantized weights, bias, and performs fake quantization on activations.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # Standard Linear layer (weights/bias will be frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Freeze parameters — weights/bias are fixed post-quantization
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False

        # Register buffers for quantization parameters
        self.register_buffer(
            "fc_scaling_factor", None
        )  # Output scaling factor (float32, shape: [out_features])

    def set_quant_params(self, weight_int, bias_int, scale):
        """
        Load quantized weights, bias, and output scaling factor.
        Inputs are assumed to be quantized (int32) and scale (float32).
        """
        with torch.no_grad():
            self.linear.weight.copy_(weight_int.to(torch.float32))
            self.linear.bias.copy_(bias_int.to(torch.float32))
        self.fc_scaling_factor = scale  # shape: (out_features,)

    def forward(self, x, act_scaling_factor=None):
        """
        Forward pass with fake quantization.
        - x is scaled down (dequantized) using act_scaling_factor
        - Output is rescaled using combined output scale
        """
        act_scaling_factor = act_scaling_factor.view(1, -1)  # shape: (1, in_features)

        # Compute combined scaling factor for output
        bias_scaling_factor = self.fc_scaling_factor * act_scaling_factor

        # Dequantize input
        x_int = torch.round(x / act_scaling_factor)  # Simulate quantized input

        # Linear transformation (float weights)
        x_out = self.linear(x_int)

        # Requantize output
        x_out = x_out * bias_scaling_factor  # Simulate quantized output

        return x_out, bias_scaling_factor


class QuantMatMul(nn.Module):
    """
    A quantized matrix multiplication module with fake quantization.
    Simulates integer matmul with scaling applied to inputs and output.
    """

    def __init__(self):
        super().__init__()

        # Output activation scaling factor (used for rescaling result)
        self.register_buffer("act_scaling_factor", None)  # shape: (out_features,)

    def set_quant_params(self, scale):
        """
        Set the output activation scaling factor.
        Typically precomputed during quant calibration.
        """
        self.act_scaling_factor = scale  # shape: (out_features,)

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        """
        Perform quantized matrix multiplication:
        - Quantize inputs using provided scaling factors
        - Multiply in float (simulated int domain)
        - Rescale the result using act_scaling_factor
        """

        # Compute combined scale for output
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B

        # Simulate quantized input (float domain)
        A_int = torch.round(A / pre_act_scaling_factor_A)  # Simulated int
        B_int = torch.round(B / pre_act_scaling_factor_B)  # Simulated int

        # Integer matrix multiplication (simulated in float)
        out_int = torch.matmul(A_int, B_int)

        # Simulate dequantization of output
        out = out_int * act_scaling_factor

        return out, act_scaling_factor


class QuantLayerNorm(nn.Module):
    """
    Inference-only integer LayerNorm using integer arithmetic and static quantization.
    All internal ops are int32, except for initial input scaling and final output scaling.
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape

        # Buffers to hold quantization parameters
        self.register_buffer("norm_scaling_factor", None)  # Final output scale (float32)
        self.register_buffer("bias_integer", None)  # Quantized bias (int32)

    def set_quant_params(self, bias_int, scale):
        """
        Set the quantized bias and final output scaling factor.
        """
        self.bias_integer = bias_int  # shape: (..., normalized_shape)
        self.norm_scaling_factor = scale  # shape: (..., normalized_shape)

    def forward(self, x, act_scaling_factor):
        """
        Args:
            x: input tensor (float32, assumed quantized: x = x_int * act_scaling_factor)
            act_scaling_factor: per-channel input scale
        Returns:
            (dequantized output tensor, output scale)
        """

        # Convert input to int32 (simulating quantized input)
        x_int = (x / act_scaling_factor).to(torch.int32)

        # Compute integer mean across last dimension
        mean_int = x_int.sum(dim=-1, keepdim=True) // x_int.shape[-1]
        y_int = x_int - mean_int  # Center input

        # Integer variance (sum of squared deviations)
        y_sq_int = y_int * y_int
        var_int = y_sq_int.sum(dim=-1, keepdim=True)

        # Approximate reciprocal std using Newton-Raphson + bitshifting
        k = torch.full_like(var_int, 1 << 16)  # Initial guess (2^16)
        for _ in range(10):
            k = (k + (var_int // k)) >> 1  # Iteratively refine k ≈ sqrt(var)

        std_int = k  # Approximate std (int32)

        # Scale factor: use fixed-point ratio to normalize
        factor = ((1 << 31) - 1) // std_int  # Fixed-point scale to normalize
        y_scaled_int = (y_int * factor) >> 1  # Normalize and shrink range

        # Add integer bias
        y_scaled_int = y_scaled_int + self.bias_integer

        # Dequantize output using learned float scale
        out = y_scaled_int * self.norm_scaling_factor  # FLOAT OP

        return out, self.norm_scaling_factor

class QuantLayerNormChannelFirst(nn.Module):
    """
    Inference-only integer LayerNorm using integer arithmetic and static quantization.
    All internal ops are int32, except for initial input scaling and final output scaling.
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape

        # Buffers to hold quantization parameters
        self.register_buffer("norm_scaling_factor", None)  # Final output scale (float32)
        self.register_buffer("bias_integer", None)  # Quantized bias (int32)

    def set_quant_params(self, bias_int, scale):
        """
        Set the quantized bias and final output scaling factor.
        """
        self.bias_integer = bias_int  # shape: (..., normalized_shape)
        self.norm_scaling_factor = scale  # shape: (..., normalized_shape)

    def forward(self, x, act_scaling_factor):
        """
        Args:
            x: input tensor (float32, assumed quantized: x = x_int * act_scaling_factor)
            act_scaling_factor: per-channel input scale
        Returns:
            (dequantized output tensor, output scale)
        """

        # Convert input to int32 (simulating quantized input)
        x_int = (x / act_scaling_factor).to(torch.int32)

        # Compute integer mean across last dimension
        mean_int = x_int.sum(dim=1, keepdim=True) // x_int.shape[1]
        y_int = x_int - mean_int  # Center input

        # Integer variance (sum of squared deviations)
        y_sq_int = y_int * y_int
        var_int = y_sq_int.sum(dim=1, keepdim=True)

        # Approximate reciprocal std using Newton-Raphson + bitshifting
        k = torch.full_like(var_int, 1 << 16)  # Initial guess (2^16)
        for _ in range(10):
            k = (k + (var_int // k)) >> 1  # Iteratively refine k ≈ sqrt(var)

        std_int = k  # Approximate std (int32)

        # Scale factor: use fixed-point ratio to normalize
        factor = ((1 << 31) - 1) // std_int  # Fixed-point scale to normalize
        y_scaled_int = (y_int * factor) >> 1  # Normalize and shrink range

        # Add integer bias
        y_scaled_int = y_scaled_int + self.bias_integer[:, None, None]

        # Dequantize output using learned float scale
        out = y_scaled_int * self.norm_scaling_factor  # FLOAT OP

        return out, self.norm_scaling_factor

class QuantSoftmax(nn.Module):
    """
    Integer-only Softmax approximation using bitshift-based exp.
    Designed for quantized inference where float ops are only used for scaling.
    """

    def __init__(self, output_bit=8, dim=-1):
        super().__init__()
        self.output_bit = output_bit  # Number of bits for output (e.g., 8-bit)
        self.dim = dim  #
        self.n = 15  # Base shift exponent for exp approximation
        self.m = 31  # Bit width for accumulator to prevent overflow

        # Buffers for quantization parameters
        self.register_buffer("act_scaling_factor", None)  # Output scale
        self.register_buffer("x0_int", None)  # Integer threshold for exp

    def set_quant_params(self, scale: float):
        """
        Sets the output scaling factor used during dequantization.
        """
        self.act_scaling_factor = torch.tensor(scale, dtype=torch.float32)
        # x0_int is deferred and lazily set during first forward pass

    def int_exp_shift(self, x_int: torch.Tensor) -> torch.Tensor:
        """
        Approximates exp(x) using integer-only bitshifts.
        This avoids floating point exponential computations.
        """
        # Pre-shift input for a tighter exp approximation
        scaled_floor = x_int >> 1  # x / 2
        scaled_exp_floor = x_int >> 4  # x / 16
        x_int = x_int + scaled_floor - scaled_exp_floor

        # Clamp lower bound to avoid overflow in negative range
        min_x_int = torch.clamp(self.n * (-self.x0_int), min=-(1 << self.m))
        x_int = torch.maximum(x_int, min_x_int)

        # Piecewise approximation of exp using quantized region
        q = x_int // (-self.x0_int)
        r = -(x_int - q * (-self.x0_int))  # Remainder within segment

        # Apply bitshift-based nonlinear scaling
        exp_int = (-r) >> 1
        exp_int = exp_int + self.x0_int
        exp_int = exp_int * (1 << (self.n - q))
        exp_int = torch.clamp(exp_int, min=0)

        return exp_int

    def forward(self, x: torch.Tensor, act_scaling_factor: torch.Tensor = None):
        """
        Forward pass of quantized softmax.
        Only floating-point operations are for rescaling inputs/outputs.
        """
        # Lazily compute x0_int from activation scale
        if self.x0_int is None:
            self.x0_int = torch.ceil(1.0 / (act_scaling_factor + 1e-6)).to(torch.int32)

        # Quantize input to int32 domain
        x_int = (x / act_scaling_factor).to(torch.int32)

        # Stabilize by subtracting max value (same as in normal softmax)
        x_int_max, _ = x_int.max(dim=self.dim, keepdim=True)
        x_int = x_int - x_int_max

        # Compute approximate exp in integer domain
        exp_int = self.int_exp_shift(x_int)

        # Integer sum of exponentials
        exp_int_sum = exp_int.sum(dim=self.dim, keepdim=True)
        exp_int_sum = torch.clamp(exp_int_sum, max=(1 << self.m) - 1)

        # Normalize using scaled factor and shift to output bitwidth
        norm_factor = (1 << self.m) - 1
        factor = norm_factor // exp_int_sum
        softmax_int = (exp_int * factor) >> (self.m - self.output_bit + 1)

        # Dequantize back to float using final output scale
        output = softmax_int * self.act_scaling_factor  # FLOAT OP
        return output, self.act_scaling_factor


class QuantGELU(nn.Module):
    """
    Integer-only approximation of GELU activation using bitshift-based sigmoid.
    All computations are performed in the integer domain, except for input/output scaling.
    """

    def __init__(self, output_bit=8):
        super().__init__()
        self.output_bit = output_bit  # Output quantization bit width

        self.n = 15  # Exponential shift base
        self.m = 31  # Accumulator bitwidth to avoid overflow

        # Buffers to store scaling and quantization parameters
        self.register_buffer("act_scaling_factor", None)  # Output scaling factor
        self.register_buffer("sigmoid_scaling_factor", None)  # Optional intermediate scaling
        self.register_buffer("x0_int", None)  # Integer scale threshold

    def set_quant_params(self, scale: float):
        """
        Sets the output scaling factor for inference.
        """
        self.act_scaling_factor = torch.tensor(scale, dtype=torch.float32)

    def int_exp_shift(self, x_int: torch.Tensor) -> torch.Tensor:
        """
        Approximates exp(x) in integer-only domain using shift-based arithmetic.
        Designed for efficient sigmoid approximation in quantized GELU.
        """
        # Shift-based approximation to reduce complexity
        scaled_floor = x_int >> 1  # x / 2
        scaled_exp_floor = x_int >> 4  # x / 16
        x_int = x_int + scaled_floor - scaled_exp_floor

        lambda_scaler = 4  # Slope of sigmoid-like region
        # Clamp minimum input to prevent overflow from very negative values
        min_val = torch.tensor(-(1 << self.m), dtype=torch.int32, device=x_int.device)
        min_x_int = torch.clamp(self.n * lambda_scaler * (-self.x0_int), min=min_val)
        x_int = torch.maximum(x_int, min_x_int)

        # Integer-based region-based exp approx
        q = x_int // (-self.x0_int)
        r = -(x_int - q * (-self.x0_int))  # Residual for piecewise

        # Approximate exp using shifts and scaled remainder
        exp_int = (-r) >> 1
        exp_int = exp_int + self.x0_int
        exp_int = exp_int << (self.n - q)

        return exp_int

    def forward(self, x: torch.Tensor, act_scaling_factor: torch.Tensor = None):
        """
        Forward pass of quantized GELU.
        Performs an integer sigmoid approximation and multiplies it with input.
        """
        # Lazy init of sigmoid input threshold (x0) based on GELU's 1.702 constant
        if self.x0_int is None:
            self.x0_int = torch.ceil(1.0 / (act_scaling_factor * 1.702 + 1e-6)).to(torch.int32)

        # Quantize input to int32
        pre_x_int = (x / act_scaling_factor).to(torch.int32)  # FLOAT OP

        # For numerical stability, subtract max
        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        # Compute sigmoid approximation via exp
        exp_int = self.int_exp_shift(x_int)
        exp_int_max = self.int_exp_shift(-x_int_max)
        exp_int_sum = exp_int + exp_int_max  # Denominator of sigmoid

        # Clamp to prevent division by zero or overflow
        exp_int_sum = torch.clamp(exp_int_sum, max=(1 << self.m) - 1, min=1)

        # Normalize sigmoid output and shift to match output_bit width
        factor = ((1 << self.m) - 1) // exp_int_sum
        sigmoid_int = (exp_int * factor) >> (self.m - self.output_bit + 1)

        # Apply GELU: x * sigmoid(x)
        x_int = pre_x_int * sigmoid_int

        # Dequantize back to float
        out = x_int * self.act_scaling_factor  # FLOAT OP
        return out, self.act_scaling_factor
