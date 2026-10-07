"""Output classes of the OmniCloudMask models, set by the labels used in training."""

CLASS_NAMES = {0: "CLEAR", 1: "THICK_CLOUD", 2: "THIN_CLOUD", 3: "CLOUD_SHADOW"}

# Classes combined into the total cloud percentage
CLOUD_CLASSES = ("THICK_CLOUD", "THIN_CLOUD")
