from .__version__ import __version__

__all__ = [
    "predict_from_load_func",
    "predict_from_array",
    "load_ls8",
    "load_multiband",
    "load_s2",
    "__version__",
]


def __getattr__(name):
    if name == "predict_from_array":
        from .cloud_mask import predict_from_array

        return predict_from_array
    elif name == "predict_from_load_func":
        from .cloud_mask import predict_from_load_func

        return predict_from_load_func
    elif name == "load_ls8":
        from .data_loaders import load_ls8

        return load_ls8
    elif name == "load_multiband":
        from .data_loaders import load_multiband

        return load_multiband
    elif name == "load_s2":
        from .data_loaders import load_s2

        return load_s2
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
