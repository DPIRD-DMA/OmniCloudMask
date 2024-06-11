from .cloud_mask import predict_from_load_func, predict_from_array
from .data_loaders import (
    load_ls8,
    load_multiband,
    load_s2,
)

__all__ = [
    "predict_from_load_func",
    "predict_from_array",
    "load_ls8",
    "load_multiband",
    "load_s2",
]
__version__ = "1.0.0"
