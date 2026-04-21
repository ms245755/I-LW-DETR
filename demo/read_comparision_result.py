import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to plot and save metrics from the CSV file
def save_metrics_to_images(csv_path, output_folder):
    # Load the CSV file into a pandas DataFrame
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Extract the block names and metrics (Cosine Similarity, Euclidean Distance, and Manhattan Distance)
    blocks = df['Block Name']
    cosine_similarity = df['Cosine Similarity']
    euclidean_distance = df['Euclidean Distance']
    manhattan_distance = df['Manhattan Distance']
    
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

# Plot vit and projector results separately.
comparison_files = {
    'vit': 'observed_tensor/comparison_results_vit.csv',
    'projector': 'observed_tensor/comparison_results_projector.csv',
}

for component, csv_path in comparison_files.items():
    output_folder = os.path.join('observed_tensor/plots', component)
    save_metrics_to_images(csv_path, output_folder)
