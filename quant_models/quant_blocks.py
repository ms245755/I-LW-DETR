import math
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, Mlp, trunc_normal_
from timm.models.vision_transformer import _load_weights

from util.box_ops import box_cxcywh_to_xyxy
from quant_models.fake_quantized.layers import *
from quant_models.fake_quantized.utils import *
from util.utils import init_weights
from torch.nn.init import constant_, xavier_uniform_
from util.decorators import save_tensor_from_pathname


def _component_set(components):
    if components is None:
        return {"linear", "conv2d", "matmul", "softmax", "gelu", "layernorm"}
    return set(components)


class IPatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=16, stride=16, padding=(0, 0), in_chans=3, embed_dim=768, num_bits=8, observe_stat=False, block_name="", quant_components=None
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.quant_components = _component_set(quant_components)
        self.quant_conv = "conv2d" in self.quant_components
        conv_cls = IViTConv2d if self.quant_conv else nn.Conv2d
        self.proj = conv_cls(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride
        ) if not self.quant_conv else conv_cls(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, weight_bit=num_bits, bias_bit=num_bits*4
        )
        self.qact_input_bridge = IViTAct(activation_bit=num_bits)
        self.qact = IViTAct(activation_bit=num_bits*2)
        self.observe_stat = observe_stat
        self.block_name = block_name

    def forward(self, x, act_scaling_factor=None):
        
        if self.quant_conv:
            if act_scaling_factor is None:
                x, act_scaling_factor = self.qact_input_bridge(x=x, act_scaling_factor=None)
            x, act_scaling_factor = self.proj(x=x, act_scaling_factor=act_scaling_factor)
        else:
            x = self.proj(x)
            act_scaling_factor = None
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Flatten patches and rearrange to (B, N, C)
        if self.quant_conv:
            act_scaling_factor = act_scaling_factor.reshape(1, 1, x.shape[2])
            x, act_scaling_factor = self.qact(x, act_scaling_factor)
        
        x = x.reshape(B, H, W, C)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"proj", out_dir="observed_tensor/quant/")

        return x, act_scaling_factor

class IFeedForward(nn.Module):
    """
    Quantized feed-forward layer (MLP block) used in Transformers.
    Includes support for different GELU approximations and optional
    observation/recording for error metrics.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout probability.
        out_dim (int, optional): Output dimension (defaults to input dim).
        observe (bool): If True, saves activation distributions to disk.
        record_rmse (bool): If True, records RMSE between FP32 and quantized activations.
        gelu_n (int): Precision parameter for GELU.
        gelu_lambda (float): Lambda scaling for GELU.
        gelu_choice (str): Choice of GELU variant ('l_shiftgelu', 'shiftgelu', 'gelu').
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout,
        out_dim=None,
        observe_gelu=False,
        record_rmse=False,
        gelu_n=23,
        gelu_lambda=6,
        gelu_choice="l_shiftgelu",
        num_bits = 8,
        quant_components=None,
        observe_stat=False,
        block_name="",
    ):
        super().__init__()
        self.quant_components = _component_set(quant_components)
        self.quant_linear = "linear" in self.quant_components
        self.quant_gelu = "gelu" in self.quant_components
        linear_cls = IViTLinear if self.quant_linear else nn.Linear
        self.fc1 = linear_cls(in_features=dim, out_features=hidden_dim, weight_bit=num_bits, bias_bit=num_bits*4) if self.quant_linear else linear_cls(dim, hidden_dim)
        self.act = ISegGELU(output_bit=16, n=gelu_n, lambda_scaler=gelu_lambda, channel_len=hidden_dim)
        self.act_fp = nn.GELU()
        self.act_no_lambda = IViTGELU(output_bit=16)
        if out_dim is None:
            out_dim = dim
        self.fc2 = linear_cls(in_features=hidden_dim, out_features=out_dim, weight_bit=num_bits, bias_bit=num_bits*4) if self.quant_linear else linear_cls(hidden_dim, out_dim)
        self.drop = nn.Dropout(p=dropout)
        self.gelu_choice = gelu_choice

        # Quantization activations around GELU and dropout
        self.qact_fc1_input_bridge = IViTAct(activation_bit=num_bits)
        self.qact_pre_gelu = IViTAct(activation_bit=num_bits, per_channel=True)
        self.qact_gelu_input_bridge = IViTAct(activation_bit=num_bits)
        self.qact_post_gelu = IViTAct(activation_bit=16)
        self.qact_fc2_input_bridge = IViTAct(activation_bit=num_bits)
        self.qact_drop2_stub = IViTAct(activation_bit=num_bits*2)
        # self.qact_post_gelu1 = IViTAct(activation_bit=16)

        # Observation flags
        self.observe_gelu = observe_gelu
        self.record_rmse = record_rmse
        self.observe_stat = observe_stat
        self.block_name = block_name
        self.ivit_rmse = 0
        self.iseg_rmse = 0

    @property
    def unwrapped(self):
        return self

    def forward(self, x, act_scaling_factor=None, block_name="Encoder_0/"):
        # First linear projection
        if self.quant_linear:
            if act_scaling_factor is None:
                x, act_scaling_factor = self.qact_fc1_input_bridge(x=x, act_scaling_factor=None)
            x, act_scaling_factor = self.fc1(x=x, act_scaling_factor=act_scaling_factor)
            x, act_scaling_factor = self.qact_pre_gelu(x=x, act_scaling_factor=act_scaling_factor)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"0_fc1", observe_stat=self.observe_stat)
        else:
            x = self.fc1(x)
            act_scaling_factor = None

        # Save activations for debugging/analysis
        if self.observe_gelu:
            x_flat = x.detach().reshape(-1)
            out_fp32 = self.act_fp(x).detach().reshape(-1)
            out_int8_lambda = self.act(x, act_scaling_factor)[0].detach().reshape(-1)
            out_int8 = self.act_no_lambda(x, act_scaling_factor)[0].detach().reshape(-1)

            save_dir = f"./visuals/layer_distributions/{block_name}"
            os.makedirs(save_dir, exist_ok=True)

            torch.save(
                {
                    "input": x_flat,
                    "output_fp32": out_fp32,
                    "output_int8_lambda": out_int8_lambda,
                    "output_int8": out_int8,
                },
                os.path.join(save_dir, "distributions.pt"),
            )

        # RMSE tracking between quantized and FP32 GELU
        if self.record_rmse:
            self.ivit_rmse += torch.mean(
                (self.act_no_lambda(x, act_scaling_factor)[0] - self.act_fp(x)) ** 2
            )
            self.iseg_rmse += torch.mean((self.act(x, act_scaling_factor)[0] - self.act_fp(x)) ** 2)

        # GELU activation choice
        if not self.quant_gelu:
            x = F.gelu(x)
            act_scaling_factor = None
        else:
            if act_scaling_factor is None:
                x, act_scaling_factor = self.qact_pre_gelu(x=x, act_scaling_factor=None)
            if self.gelu_choice == "l_shiftgelu":
                x, act_scaling_factor = self.act(x=x, act_scaling_factor=act_scaling_factor)
            elif self.gelu_choice == "shiftgelu":
                x, act_scaling_factor = self.act_no_lambda(x=x, act_scaling_factor=act_scaling_factor)
            elif self.gelu_choice == "gelu":
                x = self.act_fp(x)
                act_scaling_factor = None
            if self.quant_linear:
                x, act_scaling_factor = self.qact_post_gelu(x=x, act_scaling_factor=act_scaling_factor)
            else:
                act_scaling_factor = None
        save_tensor_from_pathname(x=x, component_path=self.block_name+"1_gelu", observe_stat=self.observe_stat)

        # Second linear projection + quantized dropout
        if self.quant_linear:
            if act_scaling_factor is None:
                x, act_scaling_factor = self.qact_post_gelu(x=x, act_scaling_factor=None)
            x, act_scaling_factor = self.fc2(x=x, act_scaling_factor=act_scaling_factor)
            x, act_scaling_factor = self.qact_drop2_stub(x=x, act_scaling_factor=act_scaling_factor)
        else:
            x = self.fc2(x)
            act_scaling_factor = None
        save_tensor_from_pathname(x=x, component_path=self.block_name+"2_fc2", observe_stat=self.observe_stat)

        return x, act_scaling_factor

class IAttention(nn.Module):
    """
    Quantized multi-head self-attention layer, CAE-compatible (q_bias/v_bias).
    """

    def __init__(self, dim, heads, dropout, use_cae=True, qkv_bias=True, num_bits: int =8, observe_stat=False, block_name="", quant_components=None, window=False):
        super().__init__()
        self.quant_components = _component_set(quant_components)
        self.quant_linear = "linear" in self.quant_components
        self.quant_matmul = "matmul" in self.quant_components
        self.quant_softmax = "softmax" in self.quant_components
        self.heads = heads
        self.dim = dim
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.attn = None
        self.window = window

        self.use_cae = use_cae

        # Linear projections for Q, K, V
        linear_cls = IViTLinear if self.quant_linear else nn.Linear
        self.proj = linear_cls(in_features=dim, out_features=dim, weight_bit=num_bits, bias_bit=num_bits*4) if self.quant_linear else linear_cls(dim, dim)

        # CAE-style separate biases (same as float Attention when use_cae=True)
        if self.use_cae:
            self.qkv  = linear_cls(in_features=dim, out_features=dim * 3, bias=False, weight_bit=num_bits, bias_bit=num_bits*4) if self.quant_linear else linear_cls(dim, dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
            # self.qact_bias = IViTAct()
        else:
            self.qkv  = linear_cls(in_features=dim, out_features=dim * 3, bias=qkv_bias, weight_bit=num_bits, bias_bit=num_bits*4) if self.quant_linear else linear_cls(dim, dim * 3, bias=qkv_bias)
            self.q_bias = None
            self.v_bias = None

        # Quantized matmul + softmax
        self.matmul = IViTMatMul()
        if window:
            self.softmax = IViTSoftmax(output_bit=16, dim=-1, m=32)
        else:
            self.softmax = IViTSoftmax(output_bit=16, dim=-1, m=32)

        # Quantized activations
        self.qact_qkv = IViTAct(activation_bit=num_bits)
        self.qact_qkv_input_bridge = IViTAct(activation_bit=num_bits)
        self.qact_q = IViTAct(activation_bit=num_bits)
        self.qact_k = IViTAct(activation_bit=num_bits)
        self.qact_v = IViTAct(activation_bit=num_bits)
        self.qact_attn_out = IViTAct(activation_bit=16)
        self.qact_attn_matmul_bridge = IViTAct(activation_bit=16)
        self.qact_qk_matmul_out = IViTAct(activation_bit=num_bits)
        self.qact_av_matmul_out = IViTAct(activation_bit=num_bits)
        self.qact_pre_proj = IViTAct(activation_bit=num_bits)
        self.qact_proj_drop = IViTAct(activation_bit=num_bits)

        self.observe_stat = observe_stat
        self.block_name = block_name

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, act_scaling_factor=None):
        B, N, C = x.shape
        assert C == self.dim

        # Project input into Q, K, V (x: [B, N, 3C])
        if self.quant_linear:
            if act_scaling_factor is None:
                x, act_scaling_factor = self.qact_qkv_input_bridge(x=x, act_scaling_factor=None)
            x, act_scaling_factor = self.qkv(x=x, act_scaling_factor=act_scaling_factor)
        else:
            x = self.qkv(x)
            act_scaling_factor = None

        # --- CAE bias injection to match float Attention(use_cae=True) ---
        # float behavior: qkv_bias = [q_bias, 0*k_bias, v_bias]
        if self.use_cae:
            # [3C]
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias),
                dim=0,
            )
            # qkv_bias, qkv_bias_scale = self.qact_bias(qkv_bias, None)
            # broadcast to [B, N, 3C]
            # x, act_scaling_factor = add_tensor_scale([(x, act_scaling_factor), (qkv_bias.view(1, 1, -1), qkv_bias_scale)])
            x = x + qkv_bias.view(1, 1, -1)
        # ---------------------------------------------------------------

        # Quantize activation after qkv (keep same pipeline)
        if self.quant_linear:
            x, act_scaling_factor_qkv = self.qact_qkv(x=x, act_scaling_factor=act_scaling_factor)
        else:
            act_scaling_factor_qkv = None

        save_tensor_from_pathname(x, self.block_name+"0_qkv", observe_stat=self.observe_stat)
        # Split Q, K, V
        qkv = x.reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        if self.quant_matmul:
            # q/k/v are [B_window, heads, tokens, head_dim].  The preceding
            # qkv projection has one concatenated scale, but Q, K, and V often
            # have different ranges after window attention reshaping and CAE
            # bias injection, so calibrate them separately for attention matmul.
            q, scale_q = self.qact_q(x=q, act_scaling_factor=None)
            k, scale_k = self.qact_k(x=k, act_scaling_factor=None)
            attn, act_scaling_factor = self.matmul(
                A=q,
                pre_act_scaling_factor_A=scale_q,
                B=k.transpose(-2, -1),
                pre_act_scaling_factor_B=scale_k,
            )
        else:
            attn = q @ k.transpose(-2, -1)
            act_scaling_factor = None

        # scale
        attn = attn * self.scale
        if act_scaling_factor is not None:
            act_scaling_factor = act_scaling_factor * self.scale
        if self.quant_matmul:
            attn, act_scaling_factor = self.qact_qk_matmul_out(
                x=attn,
                act_scaling_factor=act_scaling_factor,
            )

        # Quantize + softmax
        save_tensor_from_pathname(attn, self.block_name+"1_attn", observe_stat=self.observe_stat)
        if self.quant_softmax:
            if self.window:
                attn, act_scaling_factor = self.qact_attn_out(x=attn, act_scaling_factor=act_scaling_factor)
                attn, act_scaling_factor = self.softmax(x=attn, act_scaling_factor=act_scaling_factor)
            else:
                if act_scaling_factor is None:
                    attn, act_scaling_factor = self.qact_attn_out(x=attn, act_scaling_factor=None)
                attn, act_scaling_factor = self.softmax(
                    x=attn,
                    act_scaling_factor=act_scaling_factor,
                    dynamic_input_scale=False,
                )
        else:
            attn = F.softmax(attn, dim=-1)
            act_scaling_factor = None

        save_tensor_from_pathname(attn, self.block_name+"2_softmax", observe_stat=self.observe_stat)
        # Apply attention to values
        if self.quant_matmul:
            if act_scaling_factor is None:
                attn, act_scaling_factor = self.qact_attn_matmul_bridge(x=attn, act_scaling_factor=None)
            v, scale_v = self.qact_v(x=v, act_scaling_factor=None)
            x, act_scaling_factor = self.matmul(
                A=attn,
                pre_act_scaling_factor_A=act_scaling_factor,
                B=v,
                pre_act_scaling_factor_B=scale_v,
            )
            x, act_scaling_factor = self.qact_av_matmul_out(
                x=x,
                act_scaling_factor=act_scaling_factor,
            )
        else:
            x = attn @ v
            act_scaling_factor = None
        x = x.transpose(1, 2).reshape(B, N, C)

        # Projection + quantize
        save_tensor_from_pathname(x, self.block_name+"3_mulvalue", observe_stat=self.observe_stat)
        if self.quant_linear:
            x, act_scaling_factor = self.qact_pre_proj(x=x, act_scaling_factor=act_scaling_factor)
            x, act_scaling_factor = self.proj(x=x, act_scaling_factor=act_scaling_factor)
            x, act_scaling_factor = self.qact_proj_drop(x=x, act_scaling_factor=act_scaling_factor)
        else:
            x = self.proj(x)
            act_scaling_factor = None
        save_tensor_from_pathname(x, self.block_name+"4_output", observe_stat=self.observe_stat)
        return x, act_scaling_factor

class IBlock(nn.Module):
    """
    Transformer block consisting of:
        - LayerNorm
        - Multi-Head Attention
        - FeedForward (MLP)
    With quantized operations and residual connections.

    Args:
        dim (int): Input/Output dimension.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimension of MLP hidden layer.
        dropout (float): Dropout probability.
        drop_path (float): (Unused, always Identity here).
        gelu_n (int): Precision parameter for GELU.
        gelu_lambda (float): Lambda scaling for GELU.
        gelu_choice (str): GELU variant ('l_shiftgelu', 'shiftgelu', 'gelu').
    """

    def __init__(
        self,
        dim,
        heads,
        mlp_dim,
        dropout,
        drop_path,
        gelu_n,
        gelu_lambda,
        gelu_choice="l_shiftgelu",
        qkv_bias = True,
        observe_gelu=False,
        record_rmse=False,
        window=False,
        use_cae=True,
        num_bits=8,
        observe_stat=False,
        block_name="",
        quant_components=None,
    ):
        super().__init__()
        self.quant_components = _component_set(quant_components)
        self.quant_layernorm = "layernorm" in self.quant_components
        self.norm1 = IViTLayerNorm(normalized_shape=dim, block_name=block_name+"0_norm1/", observe_stat=observe_stat) if self.quant_layernorm else nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = IViTLayerNorm(normalized_shape=dim, block_name=block_name+"4_norm2/", observe_stat=observe_stat) if self.quant_layernorm else nn.LayerNorm(dim, eps=1e-6)
        self.attn = IAttention(dim, heads, dropout, qkv_bias=qkv_bias,num_bits=num_bits, observe_stat=observe_stat, block_name=block_name+"1_attn/", quant_components=self.quant_components, window=window)
        self.mlp = IFeedForward(
            dim, mlp_dim, dropout, observe_gelu=observe_gelu, record_rmse=record_rmse, gelu_n=gelu_n, gelu_lambda=gelu_lambda, gelu_choice=gelu_choice, num_bits=num_bits, quant_components=self.quant_components, observe_stat=observe_stat, block_name=block_name+"5_mlp/"
        )
        self.window = window

        # Quantized activations around residual paths
        self.qact_norm1_input_bridge = IViTAct(activation_bit=num_bits)
        self.pre_attn_scaler = IViTAct(activation_bit=num_bits)
        self.qact_drop_1 = IViTAct(activation_bit=num_bits*2)
        self.qact_norm2_input_bridge = IViTAct(activation_bit=num_bits)
        self.pre_mlp_scaler = IViTAct(activation_bit=num_bits)
        self.qact_drop_2 = IViTAct(activation_bit=num_bits*2)
        self.use_cae = use_cae
        if use_cae:
            init_values = 0.1
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim),requires_grad=True)
            self.qact_gamma_1 = IViTAct(activation_bit=num_bits)
            self.qact_gamma_2 = IViTAct(activation_bit=num_bits)
            self.qact_post_gamma_1 = IViTAct(activation_bit=num_bits*2)
            self.qact_post_gamma_2 = IViTAct(activation_bit=num_bits*2)
        
        self.observe_stat = observe_stat
        self.block_name = block_name

    def forward(
        self, x, mask=None, return_attention=False, act_scaling_factor=None, block_name="Encoder_0/"
    ):
        B, HW, C = x.shape
        shortcut = x
        shortcut_scale = act_scaling_factor
        if self.quant_layernorm:
            if act_scaling_factor is None:
                x, act_scaling_factor = self.qact_norm1_input_bridge(x=x, act_scaling_factor=None)
            x, act_scaling_factor = self.norm1(x, act_scaling_factor)
            x, act_scaling_factor = self.pre_attn_scaler(x=x, act_scaling_factor=act_scaling_factor)
        else:
            x = self.norm1(x)
            act_scaling_factor = None
        save_tensor_from_pathname(x=x, component_path=self.block_name+"0_norm1_output", observe_stat=self.observe_stat)
        
        if not self.window:
            x = x.reshape(B // 16, 16*HW, C)
            if mask is not None:
                mask = mask.reshape(B//16, 16*HW)

        # Attention branch
        x, act_scaling_factor = self.attn(x=x, act_scaling_factor=act_scaling_factor)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"1_attn", observe_stat=self.observe_stat)
        
        # Cong: do i need to requantize after multiply?
        if self.use_cae:
            if act_scaling_factor is None:
                x = x * self.gamma_1
            else:
                gamma_1, scale_gamma1 = self.qact_gamma_1(self.gamma_1, act_scaling_factor=None)
                x = x * gamma_1
                act_scaling_factor = act_scaling_factor * scale_gamma1
                # x, act_scaling_factor = multiply_tensor_scale(x, act_scaling_factor, gamma_1, scale_gamma1)
                x, act_scaling_factor = self.qact_post_gamma_1(x, act_scaling_factor)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"2_gamma1", observe_stat=self.observe_stat)

        if not self.window:
            x = x.reshape(B , HW, C)
            if mask is not None:
                mask = mask.reshape(B, HW)

        # Residual connection with rounding
        if act_scaling_factor is None or shortcut_scale is None:
            x = x + shortcut
            act_scaling_factor = None
        else:
            x, act_scaling_factor = add_tensor_scale(x, act_scaling_factor, shortcut, shortcut_scale)
            x, act_scaling_factor = self.qact_drop_1(x=x, act_scaling_factor=act_scaling_factor)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"3_residual1", observe_stat=self.observe_stat)
        
        shortcut = x
        shortcut_scale = act_scaling_factor

        # FeedForward branch
        if self.quant_layernorm:
            if act_scaling_factor is None:
                x, act_scaling_factor = self.qact_norm2_input_bridge(x=x, act_scaling_factor=None)
            x, act_scaling_factor = self.norm2(x=x, act_scaling_factor=act_scaling_factor)
            x, act_scaling_factor = self.pre_mlp_scaler(x=x, act_scaling_factor=act_scaling_factor)
        else:
            x = self.norm2(x)
            act_scaling_factor = None
        save_tensor_from_pathname(x=x, component_path=self.block_name+"4_norm2_output", observe_stat=self.observe_stat)

        x, act_scaling_factor = self.mlp(
            x=x, act_scaling_factor=act_scaling_factor, block_name=block_name
        )
        save_tensor_from_pathname(x=x, component_path=self.block_name+"5_mlp", observe_stat=self.observe_stat)

        if self.use_cae:
            if act_scaling_factor is None:
                x = x * self.gamma_2
            else:
                gamma_2, scale_gamma2 = self.qact_gamma_2(self.gamma_2, act_scaling_factor=None)
                x = x * gamma_2
                act_scaling_factor = act_scaling_factor * scale_gamma2
                # x, act_scaling_factor = multiply_tensor_scale(x, act_scaling_factor, gamma_2, scale_gamma2)
                x, act_scaling_factor = self.qact_post_gamma_2(x, act_scaling_factor)

            save_tensor_from_pathname(x=x, component_path=self.block_name+"6_gamma2", observe_stat=self.observe_stat)

        # Residual connection with rounding
        if act_scaling_factor is None or shortcut_scale is None:
            x = x + shortcut
            act_scaling_factor = None
        else:
            x, act_scaling_factor = add_tensor_scale(x, act_scaling_factor, shortcut, shortcut_scale)
            x, act_scaling_factor = self.qact_drop_2(x=x, act_scaling_factor=act_scaling_factor)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"7_residual2", observe_stat=self.observe_stat)

        return x, act_scaling_factor

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0

class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module"""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the "
                "dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        self._export = False

    def export(self):
        """export mode"""
        self._export = True

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        r"""
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )
        attention_weights = F.softmax(attention_weights, -1)

        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

class IMSDeformAttn(nn.Module):
    """Quantized Multi-Scale Deformable Attention Module"""

    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_float_path=False,
        value_per_channel=False,
        quant_components=None,
        observe_stat=False,
        block_name="",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the "
                "dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64
        self.d_model  = d_model
        self.n_levels = n_levels
        self.n_heads  = n_heads
        self.n_points = n_points
        self.use_float_path = use_float_path
        self.value_per_channel = value_per_channel
        self.quant_components = self._cross_attn_component_set(quant_components)
        self.quant_value_proj = "value_proj" in self.quant_components
        self.quant_sampling_offsets = "sampling_offsets" in self.quant_components
        self.quant_attention_weights = "attention_weights" in self.quant_components
        self.quant_softmax = "softmax" in self.quant_components
        self.quant_core = bool({"grid_sample_core", "weighted_sum"} & self.quant_components)
        self.quant_output_proj = "output_proj" in self.quant_components
        self.observe_stat = observe_stat
        self.block_name = block_name

        # ── quantized linear layers ────────────────────────────────────
        self.sampling_offsets  = IViTLinear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = IViTLinear(d_model, n_heads * n_levels * n_points)
        self.value_proj        = IViTLinear(d_model, d_model)
        self.output_proj       = IViTLinear(d_model, d_model)
        self.softmax           = IViTSoftmax(output_bit=16, dim=-1)

        # ── quantized activations ──────────────────────────────────────
        # one per linear output — each produces its own scale
        self.qact_input            = IViTAct()   # quantize input_flatten
        self.qact_query            = IViTAct()   # quantize query
        self.qact_value            = IViTAct(per_channel=value_per_channel)   # after value_proj
        self.qact_reference_points = IViTAct()   # quantize reference point coordinates
        self.qact_sampling_offsets = IViTAct()   # after sampling_offsets
        self.qact_sampling_locs    = IViTAct()   # quantize sampling locations
        self.qact_attn_weights     = IViTAct()   # after attention_weights + softmax
        self.qact_sampled_values   = IViTAct(per_channel=True)   # after grid sampling
        self.qact_output           = IViTAct()   # after ms_deform_attn_core

        self._reset_parameters()
        self._export = False

    def _cross_attn_component_set(self, components):
        if components is None:
            return {
                "value_proj",
                "sampling_offsets",
                "attention_weights",
                "softmax",
                "grid_sample_core",
                "weighted_sum",
                "output_proj",
            }
        components = {component for component in components if component != "none"}
        if "all" in components:
            components.update(
                {
                    "value_proj",
                    "sampling_offsets",
                    "attention_weights",
                    "softmax",
                    "grid_sample_core",
                    "weighted_sum",
                    "output_proj",
                }
            )
        if "core" in components:
            components.update({"grid_sample_core", "weighted_sum"})
        return components

    def _linear_maybe_quant(self, layer, x, scale, quant_enabled, qact_input=None):
        if quant_enabled:
            if scale is None and qact_input is not None:
                x, scale = qact_input(x, None)
            return layer(x, scale)
        return F.linear(x, layer.weight, layer.bias), None

    def export(self):
        self._export = True

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,                          # feature tensor
        reference_points,               # float coordinates — NO scale
        input_flatten,                  # feature tensor
        input_spatial_shapes,           # metadata — NO scale
        input_level_start_index,        # metadata — NO scale
        input_padding_mask=None,        # bool mask — NO scale
        query_scaling_factor=None,      # scale for query
        input_scaling_factor=None,      # scale for input_flatten
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        save_tensor_from_pathname(query, self.block_name + "0_query", observe_stat=self.observe_stat)
        save_tensor_from_pathname(reference_points, self.block_name + "1_reference_points", observe_stat=self.observe_stat)
        save_tensor_from_pathname(input_flatten, self.block_name + "2_input_flatten", observe_stat=self.observe_stat)

        if self.use_float_path:
            value = F.linear(input_flatten, self.value_proj.weight, self.value_proj.bias)
            if input_padding_mask is not None:
                value = value.masked_fill(input_padding_mask[..., None], 0)
            save_tensor_from_pathname(value, self.block_name + "3_value_proj", observe_stat=self.observe_stat)

            sampling_offsets = F.linear(query, self.sampling_offsets.weight, self.sampling_offsets.bias)
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
            attn_weights = F.linear(query, self.attention_weights.weight, self.attention_weights.bias)
            attn_weights = attn_weights.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            save_tensor_from_pathname(sampling_offsets, self.block_name + "4_sampling_offsets", observe_stat=self.observe_stat)
            save_tensor_from_pathname(attn_weights, self.block_name + "5_attention_logits", observe_stat=self.observe_stat)
            attn_weights = F.softmax(attn_weights, -1)

            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                sampling_locations = (
                    reference_points[:, :, None, :, None, :]
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                )
            elif reference_points.shape[-1] == 4:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
                )
            else:
                raise ValueError(
                    "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
                )
            save_tensor_from_pathname(sampling_locations, self.block_name + "6_sampling_locations", observe_stat=self.observe_stat)
            save_tensor_from_pathname(attn_weights, self.block_name + "7_attention_weights", observe_stat=self.observe_stat)

            value = value.transpose(1, 2).contiguous().view(
                N, self.n_heads, self.d_model // self.n_heads, Len_in
            )
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attn_weights)
            save_tensor_from_pathname(output, self.block_name + "8_core_output", observe_stat=self.observe_stat)
            output = F.linear(output, self.output_proj.weight, self.output_proj.bias)
            save_tensor_from_pathname(output, self.block_name + "9_output_proj", observe_stat=self.observe_stat)
            return self.qact_output(output, None)

        query_int, query_scale = query, query_scaling_factor
        flatten_int, flatten_scale = input_flatten, input_scaling_factor

        # ── value projection ───────────────────────────────────────────
        value, value_scale = self._linear_maybe_quant(
            self.value_proj,
            flatten_int,
            flatten_scale,
            self.quant_value_proj,
            self.qact_input,
        )
        if self.quant_value_proj or self.quant_core:
            value, value_scale = self.qact_value(value, value_scale)
        # value: (N, Len_in, d_model)

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0)
        save_tensor_from_pathname(value, self.block_name + "3_value_proj", observe_stat=self.observe_stat)

        # ── sampling offsets ───────────────────────────────────────────
        sampling_offsets, sampling_scale = self._linear_maybe_quant(
            self.sampling_offsets,
            query_int,
            query_scale,
            self.quant_sampling_offsets,
            self.qact_query,
        )
        if self.quant_sampling_offsets:
            sampling_offsets, sampling_scale = self.qact_sampling_offsets(sampling_offsets, sampling_scale)
        # dequant for view + normalizer division (non-quantizable ops)
        sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        save_tensor_from_pathname(sampling_offsets, self.block_name + "4_sampling_offsets", observe_stat=self.observe_stat)

        # ── attention weights ──────────────────────────────────────────
        attn_weights, attn_scale = self._linear_maybe_quant(
            self.attention_weights,
            query_int,
            query_scale,
            self.quant_attention_weights,
            self.qact_query,
        )
        attn_weights = attn_weights.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        save_tensor_from_pathname(attn_weights, self.block_name + "5_attention_logits", observe_stat=self.observe_stat)
        if self.quant_softmax:
            if attn_scale is None:
                attn_weights, attn_scale = self.qact_attn_weights(attn_weights, None)
            if torch.is_tensor(attn_scale) and attn_scale.numel() == self.n_heads * self.n_levels * self.n_points:
                attn_scale = attn_scale.view(self.n_heads, self.n_levels * self.n_points)
            attn_weights, attn_scale = self.softmax(attn_weights, attn_scale, dynamic_input_scale=True)
        else:
            attn_weights = F.softmax(attn_weights, -1)
            attn_scale = None
        attn_weights = attn_weights.transpose(1, 2).reshape(N * self.n_heads, Len_q, self.n_levels * self.n_points)
        if self.quant_core and attn_scale is None:
            attn_weights, attn_scale = self.qact_attn_weights(attn_weights, None)
        attn_weights = attn_weights.reshape(N, self.n_heads, Len_q, self.n_levels * self.n_points).transpose(1, 2)
        save_tensor_from_pathname(attn_weights, self.block_name + "7_attention_weights", observe_stat=self.observe_stat)

        # ── sampling locations (float op — not quantizable) ────────────
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )
        # print(sampling_locations)
        # sampling_locations, sampling_locations_scale = self.qact_sampling_locs(sampling_locations, None)
        save_tensor_from_pathname(sampling_locations, self.block_name + "6_sampling_locations", observe_stat=self.observe_stat)

        # ── deformable attention core ──────────────────────────────────
        value_float = value.transpose(1, 2).contiguous().view(
            N, self.n_heads, self.d_model // self.n_heads, Len_in
        )
        if self.quant_core:
            output = quantized_ms_deform_attn_core_pytorch(
                value_float,
                value_scale,
                input_spatial_shapes,
                sampling_locations,
                None,
                attn_weights,
                attn_scale,
                qact_sampled_values=self.qact_sampled_values,
            )
        else:
            output = ms_deform_attn_core_pytorch(value_float, input_spatial_shapes, sampling_locations, attn_weights)
            value_scale = None
        save_tensor_from_pathname(output, self.block_name + "8_core_output", observe_stat=self.observe_stat)
        output_scale = None
        if self.quant_core or self.quant_output_proj:
            output, output_scale = self.qact_output(output, None)

        # ── output projection ──────────────────────────────────────────
        output, output_scale = self._linear_maybe_quant(
            self.output_proj,
            output,
            output_scale,
            self.quant_output_proj,
            self.qact_output,
        )
        save_tensor_from_pathname(output, self.block_name + "9_output_proj", observe_stat=self.observe_stat)

        return output, output_scale

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
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


def quant_grid_sample_nearest(value, value_scale, sampling_grid):
    """Quantization-friendly nearest-neighbor sampler.

    Args:
        value: Tensor of shape [B, C, H, W], already fake-quantized.
        value_scale: Scalar, per-channel [C], or per-sample-channel [B, C]
            activation scale for `value`.
        sampling_grid: Tensor of shape [B, Len_q, P, 2] in normalized [-1, 1].

    Returns:
        Sampled tensor of shape [B, C, Len_q, P] gathered with integer-like
        nearest-neighbor indexing using the align_corners=False convention.
    """
    B, C, H, W = value.shape

    if torch.is_tensor(value_scale) and value_scale.numel() > 1:
        if value_scale.numel() == C:
            scale = value_scale.reshape(1, C, 1, 1)
        elif value_scale.numel() == B * C:
            scale = value_scale.reshape(B, C, 1, 1)
        else:
            raise RuntimeError(
                f"value_scale with {value_scale.numel()} elements cannot broadcast to value shape {tuple(value.shape)}"
            )
    elif torch.is_tensor(value_scale):
        scale = value_scale.to(device=value.device, dtype=value.dtype).reshape(1, 1, 1, 1)
    else:
        scale = value.new_tensor(value_scale).view(1, 1, 1, 1)

    value_int = round_ste.apply(value / scale)

    x_norm = sampling_grid[..., 0]
    y_norm = sampling_grid[..., 1]

    # Match grid_sample(..., align_corners=False) normalization.
    x = ((x_norm + 1.0) * W - 1.0) / 2.0
    y = ((y_norm + 1.0) * H - 1.0) / 2.0

    x_idx = round_ste.apply(x)
    y_idx = round_ste.apply(y)

    valid = (x_idx >= 0) & (x_idx <= (W - 1)) & (y_idx >= 0) & (y_idx <= (H - 1))

    x_idx = x_idx.clamp(0, W - 1).long()
    y_idx = y_idx.clamp(0, H - 1).long()

    linear_idx = (y_idx * W + x_idx).reshape(B, 1, -1).expand(-1, C, -1)
    value_flat = value_int.reshape(B, C, H * W)
    sampled_int = torch.gather(value_flat, dim=2, index=linear_idx).reshape(B, C, sampling_grid.shape[1], sampling_grid.shape[2])
    sampled_int = sampled_int * valid.unsqueeze(1).to(sampled_int.dtype)

    return sampled_int * scale


def quantized_ms_deform_attn_core_pytorch(
    value,
    value_scale,
    value_spatial_shapes,
    sampling_locations,
    sampling_locations_scale,
    attention_weights,
    attention_weights_scale,
    qact_sampled_values=None,
):
    """Fake-quantized deformable attention core.

    The geometric sampler still uses float `grid_sample`, but sampled features
    and attention weights are requantized before the weighted accumulation so
    the MAC path is executed in an INT8-emulated form.
    """
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape

    def _sampled_scale_view(scale):
        if not torch.is_tensor(scale) or scale.numel() == 1:
            return scale
        if scale.numel() == head_dim:
            return scale.view(1, head_dim, 1, 1)
        if scale.numel() == B * n_heads * head_dim:
            return scale.view(B * n_heads, head_dim, 1, 1)
        if scale.numel() == n_heads * head_dim:
            return scale.view(1, n_heads, head_dim, 1, 1).expand(B, -1, -1, -1, -1).reshape(B * n_heads, head_dim, 1, 1)
        return scale

    def _attention_scale_view(scale):
        if not torch.is_tensor(scale) or scale.numel() == 1:
            return scale
        if scale.numel() == L * P:
            return scale.view(1, 1, 1, L * P)
        if scale.numel() == n_heads * L * P:
            return scale.view(1, n_heads, 1, L * P).expand(B, -1, -1, -1).reshape(B * n_heads, 1, 1, L * P)
        return scale

    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for lid_, (H, W) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        if torch.is_tensor(value_scale) and value_scale.numel() > 1:
            value_scale_l_ = value_scale.view(n_heads, head_dim)[None].expand(B, -1, -1).reshape(B * n_heads, head_dim)
        else:
            value_scale_l_ = value_scale
        sampling_value_l_ = quant_grid_sample_nearest(value_l_, value_scale_l_, sampling_grid_l_)
        sampling_value_list.append(sampling_value_l_)

    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    if qact_sampled_values is not None:
        sampling_value_list, sampled_scale = qact_sampled_values(sampling_value_list, None)
    else:
        sampled_scale = value_scale

    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    sampled_scale = _sampled_scale_view(sampled_scale)
    attention_weights_scale = _attention_scale_view(attention_weights_scale)
    prod, prod_scale = multiply_tensor_scale(
        sampling_value_list, sampled_scale, attention_weights, attention_weights_scale
    )
    prod_int = round_ste.apply(prod / prod_scale)
    output = (prod_int * prod_scale).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()
