import torch
import numpy as np
import random
from typing import Optional, Tuple
import cv2
from collections.abc import Mapping

def make_json_serializable(obj):
    """
    Recursively convert an object into a JSON-serializable format.
    """
    if isinstance(obj, Mapping):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # fallback to string for unsupported types

def measure_distances(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distances between two sets of features.

    Args:
        features_a (torch.Tensor): A tensor of shape (D, N) where N is the number of samples and D is the feature dimension.
        features_b (torch.Tensor): A tensor of shape (D, M) where M is the number of samples and D is the feature dimension.

    Returns:
        torch.Tensor: A tensor of shape (N, M) containing the pairwise distances.
    """
    # Permute to match the expected shape for `torch.cdist`
    distances = torch.cdist(torch.permute(features_a, [1, 0]), torch.permute(features_b, [1, 0]))
    return distances


def super_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


def visualize_confidence(
    img: np.ndarray,
    pred: np.ndarray,
    truth: Optional[np.ndarray] = None,
    display_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Visualize confidence scores on an image with optional ground truth overlay.

    Args:
        img (np.ndarray): The input image (H, W, 3).
        pred (np.ndarray): The predicted confidence map (H, W).
        truth (Optional[np.ndarray]): The ground truth mask (H, W). Default is None.
        display_size (Tuple[int, int]): The size to resize the output image to (width, height).

    Returns:
        np.ndarray: The visualized image with confidence scores and optional ground truth overlay.
    """
    # Expand prediction to 3 channels and normalize
    pred = np.expand_dims(pred, axis=2)
    pred = np.repeat(pred, 3, axis=2)
    pred = np.exp(pred)  # Apply exponential scaling

    # Normalize prediction to [0, 1]
    pred = (pred - np.min(pred)) / 3  # Dividing by 3 is a normalization factor
    pred = np.clip(pred, 0, 1)

    # Apply color map to the confidence scores
    score_img = cv2.applyColorMap((pred * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend the original image with the confidence map
    img = (img.astype(np.float32) * (1 - np.sqrt(pred)) + score_img.astype(np.float32) * np.sqrt(pred)).astype(np.uint8)

    # Resize the image to the desired display size
    img = cv2.resize(img, display_size)

    # If ground truth is provided, overlay contours
    if truth is not None:
        truth = cv2.resize(truth, display_size)
        edged = cv2.Canny(truth, 30, 200)  # Canny edge detection thresholds
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)  # Draw contours in red

    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test `measure_distances`
    print("Testing `measure_distances`...")
    features_a = torch.rand(3, 5)  # 5 samples, 3 features each
    features_b = torch.rand(3, 4)  # 4 samples, 3 features each
    distances = measure_distances(features_a, features_b)
    print(f"Distances shape: {distances.shape}")
    assert distances.shape == (5, 4), "Distance computation failed."

    # Test `super_seed`
    print("Testing `super_seed`...")
    super_seed(42)
    rand1 = np.random.rand()
    super_seed(42)
    rand2 = np.random.rand()
    assert rand1 == rand2, "Seeding failed to produce reproducible results."

    # Test `visualize_confidence`
    print("Testing `visualize_confidence`...")
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # Random image
    pred = np.random.rand(256, 256)  # Random confidence map
    truth = np.zeros((256, 256), dtype=np.uint8)  # Empty ground truth
    truth[100:150, 100:150] = 255  # Add a square region as ground truth

    visualized_img = visualize_confidence(img, pred, truth)

    # Display the result using matplotlib
    plt.imshow(cv2.cvtColor(visualized_img, cv2.COLOR_BGR2RGB))
    plt.title("Visualized Confidence")
    plt.axis("off")
    plt.show()

    print("All tests passed!")