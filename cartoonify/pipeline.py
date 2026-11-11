"""Core cartoonify pipeline.

The pipeline follows the classic OpenCV cartoon effect:

1. Smooth the grayscale image with a median blur.
2. Detect edges with adaptive thresholding.
3. Flatten colors with a strong bilateral filter.
4. Combine the flattened colors with the edge mask.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CartoonifyStages:
    """Intermediate images produced by each pipeline step."""

    original: np.ndarray
    grayscale: np.ndarray
    edges: np.ndarray
    flattened: np.ndarray
    cartoon: np.ndarray


def read_image(path: str) -> np.ndarray:
    """Read an image from disk as RGB, raising ValueError if unreadable."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def detect_edges(grayscale: np.ndarray, block_size: int = 9, c: int = 9) -> np.ndarray:
    """Detect outlines with adaptive mean thresholding."""
    return cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def flatten_colors(
    image: np.ndarray, diameter: int = 9, sigma_color: int = 300, sigma_space: int = 300
) -> np.ndarray:
    """Smooth flat regions while keeping edges sharp with a bilateral filter."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def combine(flattened: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mask the flattened colors with the detected edges."""
    return cv2.bitwise_and(flattened, flattened, mask=edges)


def cartoonify(
    image: np.ndarray,
    block_size: int = 9,
    c: int = 9,
    diameter: int = 9,
    sigma_color: int = 300,
    sigma_space: int = 300,
) -> CartoonifyStages:
    """Run the full cartoonify pipeline and return every stage."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image")

    grayscale = to_grayscale(image)
    smooth_grayscale = cv2.medianBlur(grayscale, 5)
    edges = detect_edges(smooth_grayscale, block_size=block_size, c=c)
    flattened = flatten_colors(
        image, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space
    )
    cartoon = combine(flattened, edges)

    return CartoonifyStages(
        original=image,
        grayscale=grayscale,
        edges=edges,
        flattened=flattened,
        cartoon=cartoon,
    )


def save_image(image: np.ndarray, path: str) -> None:
    """Save an RGB image to disk, converting to BGR for OpenCV."""
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
