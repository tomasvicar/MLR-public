"""Classical segmentation — the baseline a learned U-Net is measured against.

The lecture says that segmentation is a special case of an image-to-image
task: a class is decided for every pixel. Before reaching for a network it
pays to find out how far the classical route gets — Otsu thresholding and
morphology. The script creates a synthetic microscopy image of nuclei with an
exactly known reference mask, runs the classical segmentation and evaluates it
with exactly the metrics that are on the slides (Dice, IoU, Tversky).

Without a network we get to Dice 0.98 on a clean image — no network is needed
there. But as soon as uneven illumination is added, the global threshold drops
to Dice 0.58, because a single number cannot hold for the whole image. That is
exactly the situation where a learned segmentation pays off: it decides from
the neighbourhood of a pixel, not from its absolute value.

    MPLBACKEND=Agg uv run python lectures/L10/code/classical_segmentation.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

N = 256


def data(background: bool, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic image of nuclei and its reference mask."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N]
    mask = np.zeros((N, N), bool)
    for _ in range(28):
        cx, cy = rng.uniform(20, N - 20, 2)
        a, b = rng.uniform(9, 16, 2)
        phi = rng.uniform(0, np.pi)
        xr = (x - cx) * np.cos(phi) + (y - cy) * np.sin(phi)
        yr = -(x - cx) * np.sin(phi) + (y - cy) * np.cos(phi)
        mask |= (xr / a) ** 2 + (yr / b) ** 2 <= 1
    image = 0.25 + 0.55 * mask
    image = ndimage.gaussian_filter(image, 1.5)
    if background:  # uneven illumination
        image += 0.95 * (x / N) ** 2 + 0.45 * (y / N)
    image += rng.normal(0, 0.05, image.shape)
    return image, mask


def otsu(image: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: maximizes the between-class variance of the histogram."""
    hist, edges = np.histogram(image, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weights1 = np.cumsum(hist)
    weights2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centres) / np.maximum(weights1, 1)
    mean2 = (np.cumsum((hist * centres)[::-1]) / np.maximum(weights2[::-1], 1))[::-1]
    between = weights1[:-1] * weights2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centres[np.argmax(between)])


def segment(image: np.ndarray) -> np.ndarray:
    threshold = otsu(image)
    mask = image > threshold
    mask = ndimage.binary_opening(mask, np.ones((5, 5)))
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))
    return ndimage.binary_fill_holes(mask)


def metrics(prediction: np.ndarray, reference: np.ndarray,
            alpha: float = 0.3, beta: float = 0.7) -> dict[str, float]:
    tp = int((prediction & reference).sum())
    fp = int((prediction & ~reference).sum())
    fn = int((~prediction & reference).sum())
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Dice": 2 * tp / (2 * tp + fp + fn),
        "IoU": tp / (tp + fp + fn),
        f"Tversky(a={alpha}, b={beta})": tp / (tp + alpha * fp + beta * fn),
    }


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for row, background in enumerate((False, True)):
        image, reference = data(background)
        prediction = segment(image)
        result = metrics(prediction, reference)
        label = "uneven background" if background else "clean image"
        print(f"\n=== {label} ===")
        for name, value in result.items():
            print(f"  {name:<22}{value:>10.3f}" if isinstance(value, float)
                  else f"  {name:<22}{value:>10,}")

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"input — {label}")
        axes[row, 1].imshow(reference, cmap="gray")
        axes[row, 1].set_title("reference mask")
        axes[row, 2].imshow(prediction, cmap="gray")
        axes[row, 2].set_title(f"Otsu + morphology, Dice = "
                               f"{result['Dice']:.3f}")
        for ax in axes[row]:
            ax.set_xticks([]), ax.set_yticks([])
    print("\nThe classical method fails as soon as the background stops behaving;")
    print("a learned segmentation copes with it, because it decides about the")
    print("context from the neighbourhood.")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
