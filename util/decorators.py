import matplotlib.pyplot as plt
import torch
import functools
import os
import inspect
from pathlib import Path
from typing import Union

from torch.cuda.amp import autocast


def amp_context(enabled):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with autocast(enabled=enabled):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_distributions(func, x, act_scaling_factor=None):
    if act_scaling_factor is None:
        result = func(x)
    else:
        result, _ = func(x, act_scaling_factor=act_scaling_factor)

    input_flat = x.detach().cpu().flatten().numpy()
    output_flat = result.detach().cpu().flatten().numpy()

    return input_flat, output_flat


def unwrapped_observer(
    input_flat,
    output_flat,
    func_name="GELU",
    save_dir: str = "./visuals/layer_distributions/",
    xlim=None,
    ylim=None,
    bins=None,
):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    os.makedirs(save_dir + func_name.split("/")[0], exist_ok=True)

    # Estimate y-max for reference
    y_max_input = np.histogram(input_flat, bins=bins, density=True)[0].max()
    y_max_output = np.histogram(output_flat, bins=bins, density=True)[0].max()

    # Set high-contrast but white-background style
    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(16, 8), dpi=150)
    plt.suptitle(f"Distribution for {func_name}", fontsize=24, fontweight="bold", color="black")

    # Output histogram
    sns.histplot(
        output_flat,
        bins=bins,
        kde=True,
        color="darkgreen",
        stat="density",
        edgecolor="black",
        linewidth=1.2,
        kde_kws={"linewidth": 2.5, "color": "black"},
    )

    plt.title("Output Distribution", fontsize=18, fontweight="bold", color="black")
    plt.xlabel("Value", fontsize=14, color="black")
    plt.ylabel("Density", fontsize=14, color="black")
    plt.xticks(color="black", fontsize=12)
    plt.yticks(color="black", fontsize=12)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    filename = os.path.join(save_dir, f"{func_name.replace(' ', '_')}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, facecolor="white")  # Ensures background stays white
    plt.close()

    return {
        "x_min": min(input_flat.min(), output_flat.min()),
        "x_max": max(input_flat.max(), output_flat.max()),
        "y_max": max(y_max_input, y_max_output),
    }


def assert_int_io(enable_assert=False):
    """
    Decorator that checks if input/output tensors are integer-like *after*
    dividing by act_scaling_factor, if provided.
    """

    def decorator(func):
        def wrapper(self, x, *args, act_scaling_factor=None, **kwargs):
            if enable_assert and act_scaling_factor is not None:
                is_integer_tensor(x / act_scaling_factor, label="Input")

            result, new_act_scaling_factor = func(
                self, x, *args, act_scaling_factor=act_scaling_factor, **kwargs
            )

            if enable_assert and new_act_scaling_factor is not None:
                is_integer_tensor(result / new_act_scaling_factor, label="Output")

            return result, new_act_scaling_factor

        return wrapper

    return decorator


def is_integer_tensor(x: torch.Tensor, tol: float = 1e-6, label="Tensor") -> bool:
    """
    Asserts tensor is integer-valued within tolerance, for debug.
    """
    if not torch.allclose(x, x.round(), atol=tol, rtol=1e-6):
        print(x)
        raise AssertionError(f"{label} is not integer-valued within tolerance.")
    return True


def plot_io_distribution(layer_name: str, save_dir: str = "./visuals/layer_distributions/"):
    """
    Decorator to plot input/output distributions of a ViT layer.

    Args:
        layer_name (str): Descriptive name like "Encoder Block 0"
        save_dir (str): Directory to save histogram figures
    """
    os.makedirs(save_dir, exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, x, *args, act_scaling_factor=None, **kwargs):
            # Detach and flatten input
            input_flat = x.detach().cpu().flatten().numpy()

            # Forward pass
            result, new_act_scaling_factor = func(
                self, x, *args, act_scaling_factor=act_scaling_factor, **kwargs
            )

            # Detach and flatten output
            output_flat = result.detach().cpu().flatten().numpy()

            # Plot histograms
            plt.figure(figsize=(10, 4))
            plt.suptitle(f"Distribution for {layer_name}", fontsize=14)

            plt.subplot(1, 2, 1)
            plt.hist(input_flat, bins=100, alpha=0.7, color="skyblue")
            plt.title("Input")

            plt.subplot(1, 2, 2)
            plt.hist(output_flat, bins=100, alpha=0.7, color="lightgreen")
            plt.title("Output")

            filename = os.path.join(save_dir, f"{layer_name.replace(' ', '_')}.png")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(filename)
            plt.close()

            return result, new_act_scaling_factor

        return wrapper

    return decorator


def track_init_args(cls):
    orig_init = cls.__init__
    sig = inspect.signature(orig_init)

    def __init__(self, *args, **kwargs):
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Store named kwargs (excluding 'self')
        self._init_kwargs = {k: v for k, v in bound_args.arguments.items() if k != "self"}

        orig_init(self, *args, **kwargs)

    cls.__init__ = __init__
    cls.get_init_kwargs = lambda self: getattr(self, "_init_kwargs", {})
    return cls
 
def save_tensor_from_pathname(
    x: torch.Tensor,
    component_path: str,
    out_dir: Union[str, Path] = "observed_tensor",
    *,
    ext: str = ".pt",
    detach: bool = True,
    to_cpu: bool = True,
    overwrite: bool = True,
    observe_stat: bool = False,
) -> Path:
    """
    component_path: like "backbone/vit/block0/attn"
    writes: <out_dir>/backbone/vit/block0/attn.pt
    """
    if not observe_stat:
        return
    # out_dir = Path(out_dir)

    # rel = Path(component_path)  # treat as relative path
    # if rel.is_absolute() or ".." in rel.parts:
    #     raise ValueError(f"component_path must be a safe relative path, got: {component_path!r}")

    file_path = Path(component_path)
    if not file_path.is_absolute() and (not file_path.parts or file_path.parts[0] != "observed_tensor"):
        file_path = Path(out_dir) / file_path
    observed_tensor_root = os.environ.get("OBSERVED_TENSOR_ROOT")
    if observed_tensor_root and file_path.parts and file_path.parts[0] == "observed_tensor":
        file_path = Path(observed_tensor_root).joinpath(*file_path.parts[1:])
    file_path = file_path.with_suffix(ext)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not overwrite:
        raise FileExistsError(str(file_path))

    if detach:
        x = x.detach()
    if to_cpu:
        x = x.cpu()

    torch.save(x, file_path)
    return file_path
