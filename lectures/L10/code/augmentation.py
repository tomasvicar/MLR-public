"""Data augmentation — nine transformations applied to one image.

The lecture only enumerates the augmentations (flip, rotation, scaling, crop,
change of brightness and contrast, noise, blurring, sharpening, erasing). Here
they are really carried out, so that it is visible what each of them does to
the image and how much the image statistics change — because that is exactly
what the network learns to ignore.

The input photograph is `grace_hopper.jpg` from matplotlib's sample data (a US
Navy photograph, public domain), so nothing is downloaded from the network.

    MPLBACKEND=Agg uv run python lectures/L10/code/augmentation.py
"""

import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SIZE = 200


def load() -> Image.Image:
    with matplotlib.cbook.get_sample_data("grace_hopper.jpg") as f:
        im = Image.open(f).convert("RGB")
    side = min(im.size)
    left = (im.size[0] - side) // 2
    return im.crop((left, 0, left + side, side)).resize(
        (SIZE, SIZE), Image.LANCZOS)


def variants(image: Image.Image) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    array = np.asarray(image).astype(float)

    enlarged = image.resize((int(SIZE * 1.35),) * 2, Image.BICUBIC)
    border = (enlarged.size[0] - SIZE) // 2
    erased = np.asarray(image).copy()
    erased[70:120, 45:105] = 127

    return {
        "original": np.asarray(image),
        "horizontal flip": np.asarray(image.transpose(Image.FLIP_LEFT_RIGHT)),
        "rotation +14 deg": np.asarray(image.rotate(14, resample=Image.BICUBIC)),
        "scaling 1.35x": np.asarray(enlarged.crop(
            (border, border, border + SIZE, border + SIZE))),
        "random crop": np.asarray(image.crop((30, 10, 140, 120)).resize(
            (SIZE, SIZE), Image.BICUBIC)),
        "brightness / contrast": np.asarray(ImageEnhance.Contrast(
            ImageEnhance.Brightness(image).enhance(1.35)).enhance(0.65)),
        "Gaussian noise": np.clip(
            array + rng.normal(0, 22, array.shape), 0, 255).astype(np.uint8),
        "blur": np.asarray(image.filter(ImageFilter.GaussianBlur(2.4))),
        "sharpening": np.asarray(image.filter(
            ImageFilter.UnsharpMask(radius=3, percent=250))),
        "random erasing": erased,
    }


def main() -> None:
    image = load()
    all_variants = variants(image)

    print(f"{'transformation':<24}{'mean':>8}{'std':>8}{'MAE vs. original':>20}")
    original = all_variants["original"].astype(float)
    for label, variant in all_variants.items():
        variant = variant.astype(float)
        mae = np.abs(variant - original).mean()
        print(f"{label:<24}{variant.mean():>8.1f}{variant.std():>8.1f}{mae:>20.1f}")
    print("\nThe geometric transformations change the MAE the most, even though")
    print("the content of the image stays the same — which is why the network")
    print("has to learn to be invariant to them.")

    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for ax, (label, variant) in zip(axes.ravel(), all_variants.items()):
        ax.imshow(variant)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Data augmentation — enlarge your dataset", fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
