import argparse
import csv
import matplotlib.pyplot as plt
import os

# Function to plot and save metrics from the CSV file
def save_metrics_to_images(csv_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    with open(csv_path, newline='') as file:
        rows = list(csv.DictReader(file))

    blocks = [row['Block Name'] for row in rows]
    cosine_similarity = [float(row['Cosine Similarity']) for row in rows]
    euclidean_distance = [float(row['Euclidean Distance']) for row in rows]
    manhattan_distance = [float(row['Manhattan Distance']) for row in rows]
    
    # Create a plot for Cosine Similarity and save it as an image
    plt.figure(figsize=(12, 6))
    plt.plot(blocks, cosine_similarity, label='Cosine Similarity', marker='o', linestyle='-', color='green')
    plt.xlabel('Block Name')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity Across Blocks')
    plt.xticks(rotation=75)  # Rotate x-axis labels by 45 degrees for better readability
    plt.legend()
    plt.tight_layout()
    cosine_image_path = f'{output_folder}/cosine_similarity.png'
    plt.savefig(cosine_image_path)
    plt.close()

    # Create a plot for Euclidean Distance and save it as an image
    plt.figure(figsize=(12, 6))
    plt.plot(blocks, euclidean_distance, label='Euclidean Distance', marker='o', linestyle='-', color='blue')
    plt.xlabel('Block Name')
    plt.ylabel('Euclidean Distance')
    plt.title('Euclidean Distance Across Blocks')
    plt.xticks(rotation=75)  # Rotate x-axis labels by 45 degrees for better readability
    plt.legend()
    plt.tight_layout()
    euclidean_image_path = f'{output_folder}/euclidean_distance.png'
    plt.savefig(euclidean_image_path)
    plt.close()

    # Create a plot for Manhattan Distance and save it as an image
    plt.figure(figsize=(12, 6))
    plt.plot(blocks, manhattan_distance, label='Manhattan Distance', marker='o', linestyle='-', color='red')
    plt.xlabel('Block Name')
    plt.ylabel('Manhattan Distance')
    plt.title('Manhattan Distance Across Blocks')
    plt.xticks(rotation=75)  # Rotate x-axis labels by 45 degrees for better readability
    plt.legend()
    plt.tight_layout()
    manhattan_image_path = f'{output_folder}/manhattan_distance.png'
    plt.savefig(manhattan_image_path)
    plt.close()

    print("Images saved in:", output_folder)

def get_args_parser():
    parser = argparse.ArgumentParser(description="Generate metric plots from tensor comparison CSVs.")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["vit", "projector"],
        choices=("vit", "projector"),
        help="Components whose comparison CSV files should be plotted.",
    )
    parser.add_argument("--observed-root", default="observed_tensor")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing comparison CSV files instead of failing.",
    )
    return parser


def main(args):
    for component in args.components:
        csv_path = os.path.join(args.observed_root, f"comparison_results_{component}.csv")
        if not os.path.isfile(csv_path):
            if args.skip_missing:
                print(f"Skipping {component}. Missing CSV: {csv_path}")
                continue
            raise FileNotFoundError(csv_path)

        output_folder = os.path.join(args.observed_root, "plots", component)
        save_metrics_to_images(csv_path, output_folder)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
