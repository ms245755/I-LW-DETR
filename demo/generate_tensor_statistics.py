import argparse
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import re

def extract_numeric_prefix(name):
    match = re.match(r'^(\d+)(?:_(.*))?$', name)
    if match:
        suffix = match.group(2) or ''
        return int(match.group(1)), suffix
    return float('inf'), name


def sort_key_from_relative_path(relative_path):
    path_without_ext = relative_path.replace('.pt', '')
    parts = path_without_ext.split(os.sep)
    return [extract_numeric_prefix(part) for part in parts]


# Define a function to load tensor files recursively from subdirectories
def load_tensor_files_recursively(folder_path, device):
    tensor_files = {}
    ordered_tensor_names = []
    tensor_paths = []
    folder_relative_paths = set()

    for root, dirs, files in os.walk(folder_path):
        dirs.sort(key=extract_numeric_prefix)
        pt_files = [filename for filename in files if filename.endswith(".pt")]
        pt_files.sort(key=extract_numeric_prefix)

        for directory_name in dirs:
            directory_relative_path = os.path.relpath(os.path.join(root, directory_name), folder_path)
            folder_relative_paths.add(directory_relative_path)

        for filename in pt_files:
            tensor_path = os.path.join(root, filename)
            relative_path = os.path.relpath(tensor_path, folder_path)
            relative_path_without_ext = relative_path.replace('.pt', '')

            # If a file has a sibling folder with the same name, prefer the folder contents.
            if relative_path_without_ext in folder_relative_paths:
                continue

            tensor_paths.append((sort_key_from_relative_path(relative_path), relative_path, tensor_path))

    tensor_paths.sort(key=lambda item: item[0])

    for _, relative_path, tensor_path in tensor_paths:
        tensor_name = relative_path.replace('.pt', '')
        ordered_tensor_names.append(tensor_name)
        tensor_files[tensor_name] = torch.load(tensor_path, map_location=device).to(device)

    return tensor_files, ordered_tensor_names

# Function to calculate statistics
def calculate_statistics(tensor):
    mean = torch.mean(tensor).item()
    median = torch.median(tensor).item()
    std = torch.std(tensor).item()
    min_val = torch.min(tensor).item()
    max_val = torch.max(tensor).item()
    return {'mean': mean, 'median': median, 'std': std, 'min': min_val, 'max': max_val}

# Function to compute cosine similarity using PyTorch
def compute_cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    cosine_sim = F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0))
    return cosine_sim.item()

# Function to calculate Euclidean distance
def compute_euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2).item()

# Function to calculate Manhattan (L1) distance
def compute_manhattan_distance(tensor1, tensor2):
    return torch.sum(torch.abs(tensor1 - tensor2)).item()

# Function to calculate normalized Euclidean distance
def compute_normalized_euclidean_distance(tensor1, tensor2):
    euclidean_dist = compute_euclidean_distance(tensor1, tensor2)
    norm1 = torch.norm(tensor1).item()
    norm2 = torch.norm(tensor2).item()
    # Normalize by the sum of magnitudes (L2 norms)
    normalized_euclidean_dist = euclidean_dist / (norm1 + norm2)
    return normalized_euclidean_dist

# Function to calculate normalized Manhattan (L1) distance
def compute_normalized_manhattan_distance(tensor1, tensor2):
    manhattan_dist = compute_manhattan_distance(tensor1, tensor2)
    norm1 = torch.sum(torch.abs(tensor1)).item()
    norm2 = torch.sum(torch.abs(tensor2)).item()
    # Normalize by the sum of L1 norms
    normalized_manhattan_dist = manhattan_dist / (norm1 + norm2)
    return normalized_manhattan_dist

# Function to plot comparison histograms and display statistics with KDE
def plot_comparison_histogram_and_statistics(tensor_float, tensor_quant, tensor_name, folder_path):
    # Plot the histograms side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Float histogram with KDE
    axes[0].hist(tensor_float.cpu().numpy().flatten(), bins=50, alpha=0.7, color='blue', density=True)
    # sns.kdeplot(tensor_float.cpu().numpy().flatten(), ax=axes[0], color='pink', linewidth=1, bw_adjust=0.5)
    axes[0].set_title(f'{tensor_name} Float Histogram')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    
    # Quant histogram with KDE
    axes[1].hist(tensor_quant.cpu().numpy().flatten(), bins=50, alpha=0.7, color='red', density=True)
    # sns.kdeplot(tensor_quant.cpu().numpy().flatten(), ax=axes[1], color='green', linewidth=1, bw_adjust=0.5)
    axes[1].set_title(f'{tensor_name} Quantized Histogram')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')

    # Calculate statistics for both
    stats_float = calculate_statistics(tensor_float)
    stats_quant = calculate_statistics(tensor_quant)
    
    # Calculate cosine similarity
    cosine_sim = compute_cosine_similarity(tensor_float, tensor_quant)
    
    # Calculate Euclidean distance
    euclidean_dist = compute_normalized_euclidean_distance(tensor_float, tensor_quant)
    
    # Calculate Manhattan distance
    manhattan_dist = compute_normalized_manhattan_distance(tensor_float, tensor_quant)
    
    # Save the comparison plot
    comparison_image_path = os.path.join(folder_path, 'comparison_histogram.png')
    plt.tight_layout()
    plt.savefig(comparison_image_path)
    plt.close()

    # Return statistics and similarity
    stats_diff = {stat: stats_float[stat] - stats_quant[stat] for stat in stats_float}
    
    return stats_float, stats_quant, stats_diff, cosine_sim, euclidean_dist, manhattan_dist

# Function to save results in the desired structure
def save_results(tensors_float, tensors_quant, folder_path):
    # Ensure the output folder exists
    os.makedirs(folder_path, exist_ok=True)

    results = {}
    for tensor_name in tensors_float:
        if tensor_name in tensors_quant:
            # Create subfolder for each tensor's block, with '/' as separator (no underscores)
            tensor_folder = os.path.join(folder_path, *tensor_name.split('/'))
            os.makedirs(tensor_folder, exist_ok=True)
            # Plot and save comparison histograms and calculate stats
            print(tensor_name)
            stats_float, stats_quant, stats_diff, cosine_sim, euclidean_dist, manhattan_dist = plot_comparison_histogram_and_statistics(
                tensors_float[tensor_name], tensors_quant[tensor_name], tensor_name, tensor_folder)
            
            # Save statistics in a text file
            stats_file_path = os.path.join(tensor_folder, 'statistics.txt')
            with open(stats_file_path, 'w') as f:
                f.write(f'Float Tensor Stats:\n')
                for stat, value in stats_float.items():
                    f.write(f'{stat}: {value:.4f}\n')
                f.write(f'\nQuantized Tensor Stats:\n')
                for stat, value in stats_quant.items():
                    f.write(f'{stat}: {value:.4f}\n')
                f.write(f'\nStatistical Differences:\n')
                for stat, value in stats_diff.items():
                    f.write(f'{stat}: {value:.4f}\n')
                f.write(f'\nCosine Similarity: {cosine_sim:.4f}\n')
                f.write(f'\nEuclidean Distance: {euclidean_dist:.4f}\n')
                f.write(f'\nManhattan Distance: {manhattan_dist:.4f}\n')
                
            results[tensor_name] = {
                'float_stats': stats_float,
                'quant_stats': stats_quant,
                'stats_diff': stats_diff,
                'cosine_similarity': cosine_sim,
                'euclidean_distance': euclidean_dist,
                'manhattan_distance': manhattan_dist
            }
    
    return results

# Function to save the comparison results to a CSV file
def save_results_to_csv(results, output_csv_path, ordered_tensor_names):
    # Prepare the data for CSV: list of rows
    rows = []

    # Iterate in the same recursive numeric order as the stored tensor files.
    for tensor_name in ordered_tensor_names:
        if tensor_name in results:
            row = [tensor_name]
            row.append(results[tensor_name]['cosine_similarity'])
            row.append(results[tensor_name]['euclidean_distance'])
            row.append(results[tensor_name]['manhattan_distance'])
            rows.append(row)
    
    # Write the rows to the CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Block Name', 'Cosine Similarity', 'Euclidean Distance', 'Manhattan Distance'])
        # Write all rows
        writer.writerows(rows)

    print(f"Results saved to {output_csv_path}")

# Function to process the results and save to CSV
def process_and_save_results(tensors_float, tensors_quant, folder_path, output_csv_path, ordered_tensor_names):
    # Process results
    results = save_results(tensors_float, tensors_quant, folder_path)
    
    # Save the results to a CSV file
    save_results_to_csv(results, output_csv_path, ordered_tensor_names)


def filter_common_tensors(tensors_float, tensors_quant, float_tensor_order, quant_tensor_order, component):
    common_names = set(tensors_float).intersection(tensors_quant)
    only_float = [name for name in float_tensor_order if name not in common_names]
    only_quant = [name for name in quant_tensor_order if name not in common_names]
    ordered_common_names = [name for name in float_tensor_order if name in common_names]

    if only_float:
        print(f"{component}: skipping {len(only_float)} float-only tensors.")
    if only_quant:
        print(f"{component}: skipping {len(only_quant)} quant-only tensors.")
    if not ordered_common_names:
        raise ValueError(f"No common float/quant tensors found for {component}.")

    return (
        {name: tensors_float[name] for name in ordered_common_names},
        {name: tensors_quant[name] for name in ordered_common_names},
        ordered_common_names,
    )


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Generate tensor statistics and per-layer histogram comparisons."
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["vit", "projector"],
        choices=("vit", "projector"),
        help="Backbone components to compare.",
    )
    parser.add_argument("--observed-root", default="observed_tensor")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip components whose float or quant tensor folder does not exist.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if float and quant tensor names do not match exactly.",
    )
    return parser


def main(args):
    float_root_folder = os.path.join(args.observed_root, "float", "backbone")
    quant_root_folder = os.path.join(args.observed_root, "quant", "backbone")
    output_folder = os.path.join(args.observed_root, "comparison")
    os.makedirs(output_folder, exist_ok=True)

    for component in args.components:
        float_folder = os.path.join(float_root_folder, component)
        quant_folder = os.path.join(quant_root_folder, component)

        if not os.path.isdir(float_folder) or not os.path.isdir(quant_folder):
            message = (
                f"Missing tensor folders for {component}: "
                f"float={float_folder}, quant={quant_folder}"
            )
            if args.skip_missing:
                print(f"Skipping {component}. {message}")
                continue
            raise FileNotFoundError(message)

        float_tensors, float_tensor_order = load_tensor_files_recursively(float_folder, args.device)
        quant_tensors, quant_tensor_order = load_tensor_files_recursively(quant_folder, args.device)

        if args.strict and float_tensor_order != quant_tensor_order:
            raise ValueError(
                f"Float and quant tensor orders do not match for {component}. "
                f"Please check {args.observed_root} folder contents."
            )
        if float_tensor_order != quant_tensor_order:
            float_tensors, quant_tensors, float_tensor_order = filter_common_tensors(
                float_tensors,
                quant_tensors,
                float_tensor_order,
                quant_tensor_order,
                component,
            )

        output_csv_path = os.path.join(args.observed_root, f"comparison_results_{component}.csv")

        process_and_save_results(
            float_tensors,
            quant_tensors,
            os.path.join(output_folder, f"backbone/{component}"),
            output_csv_path,
            float_tensor_order,
        )


if __name__ == "__main__":
    main(get_args_parser().parse_args())
