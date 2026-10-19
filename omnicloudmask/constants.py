"""Output classes of the OmniCloudMask models, set by the labels used in training."""

CLASS_NAMES = {0: "CLEAR", 1: "THICK_CLOUD", 2: "THIN_CLOUD", 3: "CLOUD_SHADOW"}

# Classes combined into the total cloud percentage
CLOUD_CLASSES = ("THICK_CLOUD", "THIN_CLOUD")

# RGBA colours written to the colour table of exported masks, so they display
# without styling. GeoTIFF colour tables can't store transparency, so clear is
# black; set value 0 transparent in the GIS to overlay a mask on imagery.
CLASS_COLOURS = {
    0: (0, 0, 0, 255),
    1: (255, 128, 13, 255),
    2: (44, 160, 44, 255),
    3: (214, 39, 40, 255),
}


def class_label(name: str) -> str:
    """Readable label for a class name, e.g. THICK_CLOUD -> Thick Cloud."""
    return name.replace("_", " ").title()
