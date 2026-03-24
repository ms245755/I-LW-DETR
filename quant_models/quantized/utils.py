from .layers import *
import inspect


def inject_real_quant_layers(module):
    """
    Replaces IViT* layers with Quant* layers in-place within a given module.
    Preserves matching weights by copying from the fake (IViT) to real (Quant) layer.
    """
    for name, child in module.named_children():
        class_name = child.__class__.__name__

        if class_name.startswith("IViT"):
            quant_class_name = class_name.replace("IViT", "Quant")
            try:
                quant_module = __import__(
                    "src.model.iseg.quantized.layers", fromlist=[quant_class_name]
                )
                quant_class = getattr(quant_module, quant_class_name)
            except (ImportError, AttributeError) as e:
                print(f"[inject_real_quant_layers] Failed to import {quant_class_name}: {e}")
                continue

            # Use named init kwargs from the tracked constructor
            try:
                init_kwargs = child.get_init_kwargs()

                # Get parameter names from quant_class.__init__, skipping 'self'
                init_sig = inspect.signature(quant_class.__init__)
                valid_params = {k: v for k, v in init_kwargs.items() if k in init_sig.parameters}

                new_layer = quant_class(**valid_params)
            except Exception as e:
                print(f"[inject_real_quant_layers] Failed to init {quant_class_name}: {e}")
                new_layer = quant_class()  # fallback

            # Copy matching parameters
            matched = {
                k: v
                for k, v in child.state_dict().items()
                if k in new_layer.state_dict() and v.shape == new_layer.state_dict()[k].shape
            }
            try:
                new_layer.load_state_dict({**new_layer.state_dict(), **matched})
            except Exception as e:
                print(f"[inject_real_quant_layers] Warning: partial state_dict load failed: {e}")

            print(
                f"[inject_real_quant_layers] Swapped {class_name} → {quant_class_name} in '{name}' ({len(matched)} params copied)"
            )

            setattr(module, name, new_layer)
        else:
            # Recurse through nested submodules
            inject_real_quant_layers(child)


def load_quant_params_from_checkpoint(model, checkpoint, device='cuda'):
    """
    Initialize QuantLinear and QuantConv2d layers in `model` with int32 weights,
    biases, and scaling loaded from checkpoint dictionary.

    Args:
        model (nn.Module): Your model containing QuantLinear and QuantConv2d layers.
        checkpoint (dict): Checkpoint containing keys like:
            - For Linear layers:
                'layer_name.weight_integer', 'layer_name.bias_integer', 'layer_name.fc_scaling_factor'
            - For Conv2d layers:
                'layer_name.weight_integer', 'layer_name.bias_integer', 'layer_name.conv_scaling_factor'
            Weights and biases are expected as int32 tensors stored as float32 tensors.
    """
    for name, module in model.named_modules():
        if hasattr(module, "set_quant_params"):
            # Common keys
            w_key = f"{name}.weight_integer"
            b_key = f"{name}.bias_integer"

            if isinstance(module, QuantLinear):
                s_key = f"{name}.fc_scaling_factor"
                if w_key in checkpoint and s_key in checkpoint:
                    weight_int = checkpoint[w_key].to(device)
                    bias_int = checkpoint[b_key].to(device) if b_key in checkpoint else None
                    fc_scale = checkpoint[s_key].to(device)
                    module.set_quant_params(weight_int, bias_int, fc_scale)
                    print(f"Loaded QuantLinear params for {name}")
                else:
                    print(f"QuantLinear params missing for {name}, skipping.")

            elif isinstance(module, QuantConv2d):
                s_key = f"{name}.conv_scaling_factor"
                if w_key in checkpoint and s_key in checkpoint:
                    weight_int = checkpoint[w_key].to(device)
                    bias_int = checkpoint[b_key].to(device) if b_key in checkpoint else None
                    conv_scale = checkpoint[s_key].to(device)
                    module.set_quant_params(weight_int, bias_int, conv_scale)
                    print(f"Loaded QuantConv2d params for {name}")
                else:
                    print(f"QuantConv2d params missing for {name}, skipping.")

            elif isinstance(module, QuantMatMul):
                s_key = f"{name}.act_scaling_factor"
                if s_key in checkpoint:
                    mul_scale = checkpoint[s_key].to(device)
                    module.set_quant_params(mul_scale)
                    print(f"Loaded QuantMatMul params for {name}")
                else:
                    print(f"QuantMatMul params missing for {name}, skipping.")

            elif isinstance(module, QuantLayerNorm):
                s_key = f"{name}.norm_scaling_factor"
                if b_key in checkpoint and s_key in checkpoint:
                    bias_int = checkpoint[b_key].to(device)
                    norm_scale = checkpoint[s_key].to(device)
                    module.set_quant_params(bias_int, norm_scale)
                    print(f"Loaded QuantLayerNorm params for {name}")
                else:
                    print(f"QuantLayerNorm params missing for {name}, skipping.")

            elif isinstance(module, QuantSoftmax):
                s_key = f"{name}.act_scaling_factor"
                if s_key in checkpoint:
                    softmax_scale = checkpoint[s_key].to(device)
                    module.set_quant_params(softmax_scale)
                    print(f"Loaded QuantSoftmax params for {name}")
                else:
                    print(f"QuantSoftmax params missing for {name}, skipping.")

            elif isinstance(module, QuantGELU):
                s_key = f"{name}.act_scaling_factor"
                if s_key in checkpoint:
                    softmax_scale = checkpoint[s_key].to(device)
                    module.set_quant_params(softmax_scale)
                    print(f"Loaded QuantGELU params for {name}")
                else:
                    print(f"QuantGELU params missing for {name}, skipping.")

            elif isinstance(module, QuantAct):
                a_key = f"{name}.act_scaling_factor"
                # o_key = f"{name}.out_scaling_factor"
                if a_key in checkpoint:
                    act_scale = checkpoint[a_key].to(device)
                    # out_scale = checkpoint[o_key].to(ptu.device)
                    module.set_quant_params(act_scale)  # , out_scale)
                    print(f"Loaded QuantAct params for {name}")
                else:
                    print(f"QuantAct params missing for {name}, skipping.")
