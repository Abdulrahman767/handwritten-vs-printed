"""Binarize document images to black ink on white background."""
from __future__ import annotations

import numpy as np
from PIL import Image


def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = float(np.dot(np.arange(256), hist))

    sum_bg = 0.0
    weight_bg = 0
    best_var = 0.0
    best_t = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t

    return best_t


def binarize_image(img: Image.Image) -> Image.Image:
    """Return a 1-channel binary PIL image: black text on white background."""
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    threshold = otsu_threshold(gray)
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def binarize_to_rgb(img: Image.Image) -> Image.Image:
    """Binary image expanded to RGB (all channels equal)."""
    return binarize_image(img).convert("RGB")


class BinarizeTransform:
    """Torchvision-compatible transform: RGB/L -> binarized RGB."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return binarize_to_rgb(img)
