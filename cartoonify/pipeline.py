"""Core cartoonify pipeline.

Four styles are available:

- ``classic``: the classic OpenCV cartoon effect (edge mask over flattened colors)
- ``sketch``: pencil-sketch look using gray-dodge
- ``comic``: posterized colors with strong outlines
- ``hdr``: painted look from repeated bilateral filtering
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CartoonifyStages:
    """Intermediate images produced by each pipeline step."""

    original: np.ndarray
    grayscale: np.ndarray
    edges: np.ndarray | None
    flattened: np.ndarray | None
    cartoon: np.ndarray


def pencil_sketch(image: np.ndarray) -> np.ndarray:
    """Create a pencil-sketch look with the gray-dodge technique."""
    grayscale = to_grayscale(image)
    inverted = 255 - grayscale
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(grayscale, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)


def posterize(image: np.ndarray, k: int = 8) -> np.ndarray:
    """Reduce the image to ``k`` dominant colors with k-means."""
    pixels = np.float32(image).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(image.shape)


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


def painted(image: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Create a painted look with repeated bilateral filtering."""
    result = image
    for _ in range(iterations):
        result = cv2.bilateralFilter(result, 9, 200, 200)
    return result


def cartoonify(
    image: np.ndarray,
    style: str = "classic",
    block_size: int = 9,
    c: int = 9,
    diameter: int = 9,
    sigma_color: int = 300,
    sigma_space: int = 300,
) -> CartoonifyStages:
    """Run the cartoonify pipeline in the given style and return every stage."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image")

    grayscale = to_grayscale(image)

    if style == "sketch":
        return CartoonifyStages(
            original=image,
            grayscale=grayscale,
            edges=None,
            flattened=None,
            cartoon=pencil_sketch(image),
        )

    if style == "hdr":
        result = painted(image)
        return CartoonifyStages(
            original=image,
            grayscale=grayscale,
            edges=None,
            flattened=result,
            cartoon=result,
        )

    smooth_grayscale = cv2.medianBlur(grayscale, 5)
    edges = detect_edges(smooth_grayscale, block_size=block_size, c=c)

    if style == "comic":
        flattened = posterize(image, k=8)
    elif style == "classic":
        flattened = flatten_colors(
            image, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space
        )
    else:
        raise ValueError(f"Unknown style: {style!r}. Use classic, sketch, comic or hdr.")

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
