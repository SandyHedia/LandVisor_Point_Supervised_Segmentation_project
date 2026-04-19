import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_results(image, sparse_mask, prediction, full_mask, save_path="result.png"):
    """
    Creates a 1x4 professional comparison for the technical report.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Panel 1: Original Satellite Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image (RGB)")
    axes[0].axis('off')

    # Panel 2: Sparse Point Input (The 'Hints')
    # We use a high-contrast colormap to make the points visible
    axes[1].imshow(image, alpha=0.5) # Background image faded
    axes[1].imshow(sparse_mask, cmap='jet', interpolation='none', alpha=1.0)
    axes[1].set_title("Input: Sparse Points")
    axes[1].axis('off')

    # Panel 3: Model Prediction (The 'Output')
    axes[2].imshow(prediction, cmap='terrain')
    axes[2].set_title("Model Prediction (Full Mask)")
    axes[2].axis('off')

    # Panel 4: Ground Truth (The 'Target')
    axes[3].imshow(full_mask, cmap='terrain')
    axes[3].set_title("Ground Truth (Target)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    