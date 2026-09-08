"""Label helpers for style (printed/handwritten) and language (ar/en)."""

CLASSES = (
    "ar_handwritten",
    "ar_printed",
    "en_handwritten",
    "en_printed",
)

STYLE_CLASSES = ("handwritten", "printed")

STYLE_TO_IDX = {name: i for i, name in enumerate(STYLE_CLASSES)}


def style_from_label(label: str) -> str:
    """Extract printed/handwritten from a 4-class folder label."""
    return label.rsplit("_", 1)[-1]


def make_label(language: str, style: str) -> str:
    language = language.lower()
    style = style.lower()
    if language not in {"ar", "en"}:
        raise ValueError(f"Unsupported language: {language}")
    if style not in {"handwritten", "printed"}:
        raise ValueError(f"Unsupported style: {style}")
    return f"{language}_{style}"


def parse_label(label: str) -> dict[str, str]:
    language, style = label.split("_", 1)
    return {"language": language, "style": style, "label": label}


def infer_metadata_from_name(filename: str, parent_style: str | None = None) -> tuple[str, str]:
    """Guess language/style from dataset filename conventions."""
    name = filename.lower()
    style = parent_style or ("handwritten" if "handwritten" in name else "printed")

    if (
        name.startswith("iam_")
        or "_en_" in name
        or name.startswith("synth_en")
        or name.startswith("printed_en_")
        or name.startswith("real_printed_en_")
        or name.startswith("handwritten_en_")
        or name.startswith("notebook_en_")
        or name.startswith("page_en_")
    ):
        language = "en"
    elif (
        name.startswith("khatt_")
        or name.startswith("muharaf_")
        or name.startswith("printed_ar_")
        or name.startswith("real_printed_ar_")
        or name.startswith("page_ar_")
        or "_ar_" in name
        or name.startswith("synth_ar")
    ):
        language = "ar"
    elif "handwritten_ledger" in name or "receipt" in name or "invoice" in name:
        language = "ar"
        if parent_style is None:
            style = "handwritten" if "ledger" in name else "printed"
    else:
        language = "ar"

    return language, style
