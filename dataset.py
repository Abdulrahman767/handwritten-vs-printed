"""Dataset helpers for binarized text-region crops."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from binarize import BinarizeTransform
from labels import STYLE_TO_IDX, style_from_label

IMG_SIZE = 224
# Minimum crop footprint before letterbox — matches tiny invoice line crops on held-out docs.
MIN_CROP_H = 32
MIN_CROP_W = 120
_binarize = BinarizeTransform()


class LetterboxPadResize:
    """Upscale tiny crops, pad to square, then resize — consistent input for all sources.

    Wide invoice lines get min_w; compact handwriting fragments only get min_h so
    narrow crops are not stretched into printed-looking bars.
    """

    def __init__(
        self,
        size: int = IMG_SIZE,
        min_h: int = MIN_CROP_H,
        min_w: int = MIN_CROP_W,
        wide_aspect: float = 2.5,
        fill: int = 255,
    ):
        self.size = size
        self.min_h = min_h
        self.min_w = min_w
        self.wide_aspect = wide_aspect
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        scale = self.min_h / max(min(w, h), 1)
        if w / max(h, 1) >= self.wide_aspect and w < self.min_w:
            scale = max(scale, self.min_w / max(w, 1))
        if h / max(w, 1) >= self.wide_aspect and h < self.min_w:
            scale = max(scale, self.min_w / max(h, 1))
        scale = max(scale, 1.0)
        if scale > 1.0:
            w = max(1, int(w * scale))
            h = max(1, int(h * scale))
            # Sharp upscale for binarized strokes — LANCZOS blurs tiny HW into printed-looking edges.
            img = img.resize((w, h), Image.Resampling.NEAREST)

        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (self.fill, self.fill, self.fill))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas.resize((self.size, self.size), Image.Resampling.NEAREST)


class RandomRuledLines:
    """Simulate notebook ruling — common on real handwritten pages."""

    def __init__(self, p: float = 0.35):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        out = img.convert("RGB")
        draw = ImageDraw.Draw(out)
        w, h = out.size
        gap = random.randint(22, 34)
        color = random.randint(150, 200)
        for y in range(gap, h, gap):
            draw.line([(0, y), (w, y)], fill=(color, color, color), width=1)
        return out


class RandomGaussianBlur:
    def __init__(self, p: float = 0.35):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))


class RandomGaussianNoise:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        out = img.convert("RGB")
        noise = Image.effect_noise(out.size, random.uniform(6, 14))
        return Image.blend(out, noise.convert("RGB"), alpha=random.uniform(0.04, 0.12))


class RandomDownscale:
    """Shrink large training crops to invoice-like sizes (held-out avg ~77x33)."""

    def __init__(self, p: float = 0.55, scale_range: tuple[float, float] = (0.06, 0.35)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        nw = max(12, int(w * scale))
        nh = max(8, int(h * scale))
        return img.resize((nw, nh), Image.Resampling.LANCZOS)


class PathAwareTrainTransform:
    """Apply stronger downscale on handwritten crops (match tiny held-out regions)."""

    def __init__(self, base: transforms.Compose):
        self.base = base
        self.hw_downscale = RandomDownscale(p=0.90, scale_range=(0.03, 0.18))
        self.default_downscale = RandomDownscale(p=0.55, scale_range=(0.06, 0.35))

    def __call__(self, img: Image.Image, path: str = "") -> torch.Tensor:
        parent = Path(path).parent.name if path else ""
        style = style_from_label(parent) if parent else ""
        if style == "handwritten":
            img = self.hw_downscale(img)
        else:
            img = self.default_downscale(img)
        return self.base(img)


class RandomContrast:
    def __init__(self, p: float = 0.3, spread: tuple[float, float] = (0.88, 1.12)):
        self.p = p
        self.spread = spread

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        factor = random.uniform(*self.spread)
        return ImageEnhance.Contrast(img).enhance(factor)


def _build_train_post_downscale() -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        _binarize,
        RandomRuledLines(p=0.35),
        LetterboxPadResize(),
        transforms.RandomRotation(6),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.94, 1.06)),
        RandomGaussianBlur(p=0.25),
        RandomGaussianNoise(p=0.2),
        RandomContrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_train_transforms() -> PathAwareTrainTransform:
    return PathAwareTrainTransform(_build_train_post_downscale())


eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    _binarize,
    LetterboxPadResize(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class RemappedImageFolder(torch.utils.data.Dataset):
    """ImageFolder with labels mapped to a fixed class_to_idx dict."""

    def __init__(
        self,
        root: str,
        class_to_idx: dict[str, int],
        transform,
        *,
        style_only: bool = False,
    ):
        self.inner = datasets.ImageFolder(root, transform=transform)
        self.class_to_idx = class_to_idx
        self.style_only = style_only
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        path, _ = self.inner.samples[index]
        img = self.inner.loader(path)
        if self.inner.transform:
            img = self.inner.transform(img)
        label_name = Path(path).parent.name
        if self.style_only:
            label_name = style_from_label(label_name)
        return img, self.class_to_idx[label_name]


class StyleImageFolder(torch.utils.data.Dataset):
    """ImageFolder that maps 4-class folders to printed vs handwritten."""

    def __init__(self, root: str, transform):
        self.inner = datasets.ImageFolder(root, transform=transform)
        self.class_to_idx = dict(STYLE_TO_IDX)
        self.classes = list(STYLE_TO_IDX.keys())

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        path, _ = self.inner.samples[index]
        img = self.inner.loader(path)
        if self.inner.transform:
            if isinstance(self.inner.transform, PathAwareTrainTransform):
                img = self.inner.transform(img, path)
            else:
                img = self.inner.transform(img)
        style = style_from_label(Path(path).parent.name)
        return img, self.class_to_idx[style]


def _balanced_sampler(dataset) -> WeightedRandomSampler:
    if hasattr(dataset, "inner"):
        targets = [
            dataset.class_to_idx[style_from_label(Path(path).parent.name)]
            for path, _label in dataset.inner.samples
        ]
    else:
        targets = [label for _path, label in dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets), minlength=len(dataset.classes))
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    sample_weights = class_weights[torch.tensor(targets)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _print_class_counts(root: Path, split: str) -> None:
    split_dir = root / split
    if not split_dir.exists():
        return
    print(f"  {split} class counts:")
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.glob("*.png")))
            print(f"    {class_dir.name}: {n:,}")


def _is_real_printed_en(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("sroie", "funsd", "donkey12"))


def _is_synthetic_printed(name: str, class_name: str) -> bool:
    lowered = name.lower()
    if class_name == "en_printed":
        if _is_real_printed_en(lowered):
            return False
        return lowered.startswith(("printed_en_", "real_printed_en_"))
    if class_name == "ar_printed":
        return lowered.startswith(("printed_ar_", "real_printed_ar_"))
    return False


def filter_synthetic_printed(
    samples: list[tuple[str, int]],
    classes: list[str],
    *,
    max_synthetic_en_printed: int,
    max_synthetic_ar_printed: int,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Cap font-rendered printed crops so the model sees more real document styles."""
    keep: list[tuple[str, int]] = []
    synth_en: list[tuple[str, int]] = []
    synth_ar: list[tuple[str, int]] = []

    for path, label in samples:
        class_name = classes[label]
        name = Path(path).name
        if _is_synthetic_printed(name, class_name):
            if class_name == "en_printed":
                synth_en.append((path, label))
            else:
                synth_ar.append((path, label))
        else:
            keep.append((path, label))

    rng = random.Random(seed)
    rng.shuffle(synth_en)
    rng.shuffle(synth_ar)
    keep.extend(synth_en[:max_synthetic_en_printed])
    keep.extend(synth_ar[:max_synthetic_ar_printed])
    rng.shuffle(keep)
    return keep


class FilteredImageFolder(datasets.ImageFolder):
    """ImageFolder with an optional post-load sample filter."""

    def __init__(self, root: str, transform=None, *, filter_fn=None):
        super().__init__(root, transform=transform)
        if filter_fn is not None:
            self.samples = filter_fn(self.samples, self.classes)
            self.targets = [label for _path, label in self.samples]
            self.imgs = self.samples


def compute_style_class_weights(
    dataset,
    *,
    handwritten_mult: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency weights from natural distribution (use with shuffle, not oversampling)."""
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    if hasattr(dataset, "inner"):
        samples = dataset.inner.samples
        mapping = dataset.class_to_idx
        for path, _label in samples:
            style = style_from_label(Path(path).parent.name)
            counts[mapping[style]] += 1
    else:
        for _path, label in dataset.samples:
            counts[label] += 1
    weights = (counts.sum() / (len(counts) * counts.clamp(min=1))).float()
    if handwritten_mult != 1.0 and "handwritten" in STYLE_TO_IDX:
        weights[STYLE_TO_IDX["handwritten"]] *= handwritten_mult
    return weights


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    *,
    balance_classes: bool = False,
    max_synthetic_en_printed: int = 0,
    max_synthetic_ar_printed: int = 0,
    style_only: bool = True,
    handwritten_weight_mult: float = 1.0,
) -> tuple[DataLoader, DataLoader, dict, torch.Tensor]:
    data_root = Path(data_dir)

    def train_filter(samples, classes):
        if max_synthetic_en_printed <= 0 and max_synthetic_ar_printed <= 0:
            return samples
        before = len(samples)
        filtered = filter_synthetic_printed(
            samples,
            classes,
            max_synthetic_en_printed=max_synthetic_en_printed,
            max_synthetic_ar_printed=max_synthetic_ar_printed,
        )
        dropped = before - len(filtered)
        if dropped:
            print(
                f"  capped synthetic printed: dropped {dropped:,} crops "
                f"(max en={max_synthetic_en_printed:,}, ar={max_synthetic_ar_printed:,})"
            )
        return filtered

    if style_only:
        train_ds = StyleImageFolder(
            str(data_root / "train"),
            transform=build_train_transforms(),
        )
        filtered = train_filter(train_ds.inner.samples, train_ds.inner.classes)
        train_ds.inner.samples = filtered
        train_ds.inner.targets = [label for _path, label in filtered]
        train_ds.inner.imgs = filtered

        val_ds = StyleImageFolder(str(data_root / "val"), transform=eval_transforms)
        class_to_idx = dict(STYLE_TO_IDX)
    else:
        train_ds = FilteredImageFolder(
            str(data_root / "train"),
            transform=build_train_transforms(),
            filter_fn=train_filter,
        )
        val_ds = datasets.ImageFolder(str(data_root / "val"), transform=eval_transforms)
        class_to_idx = train_ds.class_to_idx

    train_sampler = _balanced_sampler(train_ds) if balance_classes else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    class_weights = compute_style_class_weights(
        train_ds,
        handwritten_mult=handwritten_weight_mult,
    )
    return train_loader, val_loader, class_to_idx, class_weights


def get_held_out_loader(
    held_out_dir: str,
    class_to_idx: dict[str, int],
    batch_size: int = 32,
    num_workers: int = 0,
    *,
    style_only: bool = True,
) -> DataLoader | None:
    """Loader for real-document held-out crops (val/ subfolder)."""
    val_dir = Path(held_out_dir) / "val"
    if not val_dir.is_dir() or not any(val_dir.iterdir()):
        return None
    if style_only:
        ds = StyleImageFolder(str(val_dir), eval_transforms)
    else:
        ds = RemappedImageFolder(str(val_dir), class_to_idx, eval_transforms)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
