import numpy as np
import torch
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

def apply_histogram_matching(image, reference_path):
    """
    Stabilizes satellite imagery style by matching the histogram of the 
    input image to a 'Gold Standard' reference image.
    
    Args:
        image (np.ndarray): Input image [H, W, 3]
        reference_path (str): Path to the reference image file.
    """
    try:
        reference = plt.imread(reference_path)
        # We match per channel to preserve color relationships
        matched = match_histograms(image, reference, channel_axis=-1)
        return np.clip(matched, 0, 255).astype(np.uint8)
    except FileNotFoundError:
        print(f"Warning: Reference image not found at {reference_path}. Skipping matching.")
        return image

def calculate_miou(preds, targets, num_classes, ignore_index=-1):
    """
    Calculates Mean Intersection over Union (mIoU).
    This is the industry-standard metric for segmentation.
    """
    ious = []
    # Convert to numpy for easier logical indexing
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    for cls in range(num_classes):
        # We only calculate IoU for pixels that are NOT the ignore_index
        # in the ground truth.
        intersection = np.logical_and((preds == cls), (targets == cls)).sum()
        union = np.logical_and(
            np.logical_or((preds == cls), (targets == cls)), 
            (targets != ignore_index)
        ).sum()

        if union == 0:
            continue  # Avoid division by zero if class is absent
            
        ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0

def save_checkpoint(state, filename="checkpoint.pth"):
    """Saves the model state for reproducibility."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")
