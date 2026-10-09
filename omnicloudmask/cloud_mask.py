import os
import warnings
from collections import deque
from contextlib import AbstractContextManager, closing, nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Union,
    overload,
)

import numpy as np
import torch
from rasterio.profiles import Profile
from tqdm.auto import tqdm

from .__version__ import __version__
from .download_models import get_models
from .model_utils import (
    channel_norm_torch,
    create_gradient_mask,
    default_device,
    get_torch_dtype,
    infer_batch,
    inference_and_store,
    load_model_from_weights,
    compile_torch_model,
)
from .mps_patch import _get_model_in_channels, patch_models_for_mps
from .optimizations import optimized_argmax
from .raster_utils import (
    batch_nodata_mask,
    get_patch,
    make_patch_indexes,
    mask_prediction,
    save_prediction,
)


# PyTorch's MPS backend shares one command buffer per process, so GPU work encoded
# from two threads at once fails a Metal assertion and aborts the process.
_MPS_LOCK = RLock()


def device_lock(*devices: torch.device) -> AbstractContextManager:
    """Serialise work across threads when any of the devices is MPS. Other devices
    don't need it."""
    if any(device.type == "mps" for device in devices):
        return _MPS_LOCK
    return nullcontext()


def locked_coordinator(**kwargs) -> Optional[np.ndarray]:
    """Run coordinator while holding the device lock for its devices."""
    with device_lock(kwargs["inference_device"], kwargs["mosaic_device"]):
        return coordinator(**kwargs)


def warn_cpu_reduced_precision(device: torch.device, dtype: torch.dtype) -> None:
    """Warn if using CPU with reduced precision (fp16/bf16) as it degrades
    performance."""
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype_name = "fp16" if dtype == torch.float16 else "bf16"
        warnings.warn(
            f"Using CPU inference with {dtype_name} precision may "
            f"significantly degrade performance. "
            f"Consider using fp32 (float32) precision for CPU inference "
            f"instead. Set inference_dtype='fp32' or "
            f"inference_dtype=torch.float32.",
            UserWarning,
            stacklevel=3,
        )


def compile_batches(
    batch_size: int,
    patch_size: int,
    patch_indexes: list[tuple[int, int, int, int]],
    input_array: np.ndarray,
    no_data_value: int | float,
    inference_device: torch.device,
    inference_dtype: torch.dtype,
) -> Generator[tuple[torch.Tensor, list[tuple[int, int, int, int]]], None, None]:
    """Compile batches of patches with queue-based processing.

    Queue limits memory to 1 batch worth of results. Always yields full batches
    except the final one which may be partial.
    """

    # Queue blocks when full, preventing memory buildup
    result_queue = Queue(maxsize=batch_size)

    def worker(idx: tuple[int, int, int, int]) -> None:
        """Extract patch and put result in queue. May put (None, None) for invalid patches."""  # noqa: E501
        try:
            result_queue.put(get_patch(input_array, idx, no_data_value))
        except Exception as e:
            # pass errors to the consumer, otherwise it waits forever for a result
            result_queue.put(e)

    executor = ThreadPoolExecutor(max_workers=batch_size)
    # Submit all work upfront - queue maxsize provides limiting
    futures = [executor.submit(worker, idx) for idx in patch_indexes]

    try:
        # Collect valid results into full batches
        all_indexes = set()
        index_batch = []
        patch_batch_array = np.zeros(
            (batch_size, input_array.shape[0], patch_size, patch_size), dtype=np.float32
        )

        for i in range(len(patch_indexes)):
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result
            patch, new_index = result

            # Skip invalid patches (get_patch returned None)
            if patch is not None and new_index not in all_indexes:
                index_batch.append(new_index)
                patch_batch_array[len(index_batch) - 1] = patch
                all_indexes.add(new_index)

            # Yield full batch or final partial batch
            is_last_task = i == len(patch_indexes) - 1
            if len(index_batch) == batch_size or (index_batch and is_last_task):
                input_tensor = torch.as_tensor(
                    patch_batch_array[: len(index_batch)], dtype=torch.float32
                )
                if inference_device.type == "cuda":
                    input_tensor = input_tensor.pin_memory()
                    input_tensor = input_tensor.to(
                        device=inference_device,
                        dtype=inference_dtype,
                        non_blocking=True,
                    )
                else:
                    input_tensor = input_tensor.to(
                        device=inference_device,
                        dtype=inference_dtype,
                    )

                yield input_tensor, index_batch
                index_batch = []
    finally:
        # If stopped early (e.g. inference raised an error), cancel queued work and
        # drain the queue so workers blocked on a full queue can finish
        for future in futures:
            future.cancel()
        while not all(future.done() for future in futures):
            try:
                result_queue.get(timeout=0.01)
            except Empty:
                pass
        executor.shutdown(wait=True)


def run_models_on_array(
    models: list[torch.nn.Module],
    input_array: np.ndarray,
    pred_tracker: torch.Tensor,
    grad_tracker: Union[torch.Tensor, None],
    patch_size: int,
    patch_overlap: int,
    inference_device: torch.device,
    batch_size: int = 2,
    inference_dtype: torch.dtype = torch.float32,
    no_data_value: int | float = 0,
) -> None:
    """Used to execute the model on the input array, in patches. Predictions are stored
    in pred_tracker and grad_tracker, updated in place."""
    patch_indexes = make_patch_indexes(
        array_height=input_array.shape[1],
        array_width=input_array.shape[2],
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    gradient = create_gradient_mask(
        patch_size, patch_overlap, device=inference_device, dtype=inference_dtype
    )

    input_tensor_gen = compile_batches(
        batch_size=batch_size,
        patch_size=patch_size,
        patch_indexes=patch_indexes,
        input_array=input_array,
        no_data_value=no_data_value,
        inference_device=inference_device,
        inference_dtype=inference_dtype,
    )

    for patch_batch, index_batch in input_tensor_gen:
        inference_and_store(
            models=models,
            patch_batch=patch_batch,
            index_batch=index_batch,
            pred_tracker=pred_tracker,
            gradient=gradient,
            grad_tracker=grad_tracker,
        )


# ideally the patch size would be above SOFT_MINIMUM_PATCH_SIZE
SOFT_MINIMUM_PATCH_SIZE = 50

# if the patch size is below HARD_MINIMUM_PATCH_SIZE the models will error
HARD_MINIMUM_PATCH_SIZE = 32


def validate_input_shape(shape: tuple[int, ...]) -> None:
    """Check an input image has 3 dimensions and is large enough for the models."""
    if len(shape) != 3:
        raise ValueError(
            f"Input array must have 3 dimensions, found {len(shape)}. "
            f"The input should be in format (bands (red,green,NIR), height, width)."
        )

    # check the width and height are greater than or equal to HARD_MINIMUM_PATCH_SIZE
    if min(shape[1], shape[2]) < HARD_MINIMUM_PATCH_SIZE:
        raise ValueError(
            f"Input array must have a width and height greater than or "
            f"equal to {HARD_MINIMUM_PATCH_SIZE} pixels, "
            f"found shape {tuple(shape)}. "
            f"You may add a nodata buffer to pad the input array to the minimum size. "
            f"The input should be in format (bands (red,green,NIR), height, width)."
        )
    if min(shape[1], shape[2]) < SOFT_MINIMUM_PATCH_SIZE:
        warnings.warn(
            f"Input width or height is less than {SOFT_MINIMUM_PATCH_SIZE} pixels, "
            f"found shape {tuple(shape)}. Small image may not provide adequate "
            f"spatial context for the model.",
            stacklevel=3,
        )


def check_patch_size(
    input_array: np.ndarray,
    no_data_value: int | float,
    patch_size: int,
    patch_overlap: int,
) -> tuple[int, int]:
    """Used to check the inputs and adjust the patch size and overlap if necessary."""
    validate_input_shape(input_array.shape)

    # if the input has a lot of no data values and the patch size is larger than
    # half the image size, we reduce the patch size and overlap
    if np.count_nonzero(input_array == no_data_value) / input_array.size > 0.3:
        if patch_size > min(input_array.shape[1], input_array.shape[2]) / 2:
            patch_size = max(
                min(input_array.shape[1], input_array.shape[2]) // 2,
                HARD_MINIMUM_PATCH_SIZE,
            )  # make sure the new size is at least the hard minimum
            if patch_size // 2 < patch_overlap:
                patch_overlap = patch_size // 2

            warnings.warn(
                f"Significant no-data areas detected. Adjusting patch size "
                f"to {patch_size}px and overlap to {patch_overlap}px to minimize "
                f"no-data patches.",
                stacklevel=2,
            )

    # if the patch size is larger than the image size,
    # we reduce the patch size and overlap
    if patch_size > min(input_array.shape[1], input_array.shape[2]):
        patch_size = max(
            min(input_array.shape[1], input_array.shape[2]), HARD_MINIMUM_PATCH_SIZE
        )  # make sure the new size is at least the hard minimum

        if patch_size // 2 < patch_overlap:
            patch_overlap = patch_size // 2
        warnings.warn(
            f"Patch size too large, reducing to {patch_size} and "
            f"overlap to {patch_overlap}.",
            stacklevel=2,
        )

    # if the patch overlap is larger than the patch size, raise an error
    if patch_overlap >= patch_size:
        raise ValueError(
            f"Patch overlap {patch_overlap}px must be less than patch size "
            f"{patch_size}px."
        )

    if patch_size < HARD_MINIMUM_PATCH_SIZE:
        raise ValueError(
            f"Patch size {patch_size}px must be at least {HARD_MINIMUM_PATCH_SIZE}px."
        )
    if patch_size < SOFT_MINIMUM_PATCH_SIZE:
        warnings.warn(
            f"Patch size {patch_size}px is less than {SOFT_MINIMUM_PATCH_SIZE}px. "
            "Small patch sizes may not provide adequate spatial context for the model.",
            stacklevel=2,
        )
    return patch_overlap, patch_size


def postprocess_preds(
    preds: torch.Tensor,
    export_confidence: bool,
    softmax_output: bool,
    class_dim: int = 0,
) -> torch.Tensor:
    """Convert raw model predictions into float32 confidence values or uint8 class
    indices along class_dim, on the same device as preds."""
    if export_confidence:
        if softmax_output:
            preds = torch.clip(
                (torch.nn.functional.softmax(preds, class_dim) + 0.001),
                0.001,
                0.999,
            )
            # replace nan with 0, for areas with no predictions
            preds = torch.nan_to_num(preds, nan=0.0)
        return preds.float()

    # Use optimized argmax (pairwise for CPU/MPS, standard for CUDA)
    return optimized_argmax(preds, dim=class_dim, keepdim=True).to(dtype=torch.uint8)


def build_export_profile(profile: Optional[Profile], pred: np.ndarray) -> Profile:
    """Build a GeoTIFF export profile for a (bands, height, width) prediction.
    The profile's height and width are kept if present, as loaders that resample
    return the profile of the source grid, and the prediction is written onto it."""
    if profile is None:
        profile = Profile()
    export_profile = profile.copy()
    export_profile.setdefault("height", pred.shape[1])
    export_profile.setdefault("width", pred.shape[2])
    export_profile.update(
        dtype=pred.dtype,
        count=pred.shape[0],
        compress="lzw",
        nodata=None,
        driver="GTiff",
    )
    return export_profile


def make_output_path(
    scene_path: Union[Path, str], output_dir: Optional[Union[Path, str]]
) -> Path:
    """Return the prediction file path for a scene, next to the scene if
    output_dir is None."""
    scene_path = Path(scene_path)
    file_name = f"{scene_path.stem}_OCM_v{__version__.replace('.', '_')}.tif"

    if output_dir is None:
        return scene_path.parent / file_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return Path(output_dir) / file_name


def coordinator(
    input_array: np.ndarray,
    models: list[torch.nn.Module],
    inference_dtype: torch.dtype,
    export_confidence: bool,
    softmax_output: bool,
    inference_device: torch.device,
    mosaic_device: torch.device,
    patch_size: int,
    patch_overlap: int,
    batch_size: int,
    profile: Optional[Profile] = None,
    output_path: Path = Path(""),
    no_data_value: int | float = 0,
    pbar: Optional[tqdm] = None,
    apply_no_data_mask: bool = False,
    export_to_disk: bool = True,
    save_executor: Optional[ThreadPoolExecutor] = None,
    pred_classes: int = 4,
    save_futures: Optional[list[Future]] = None,
) -> np.ndarray:
    """Used to coordinate the process of predicting from an input array.
    If save_futures is provided, futures for saves submitted to save_executor are
    appended to it so the caller can check them for errors."""

    patch_overlap, patch_size = check_patch_size(
        input_array, no_data_value, patch_size, patch_overlap
    )

    pred_tracker = torch.zeros(
        (pred_classes, *input_array.shape[1:3]),
        dtype=inference_dtype,
        device=mosaic_device,
    )

    grad_tracker = (
        torch.zeros(input_array.shape[1:3], dtype=inference_dtype, device=mosaic_device)
        if export_confidence
        else None
    )

    run_models_on_array(
        models=models,
        input_array=input_array,
        pred_tracker=pred_tracker,
        grad_tracker=grad_tracker,
        inference_device=inference_device,
        inference_dtype=inference_dtype,
        no_data_value=no_data_value,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
    )

    if export_confidence:
        if grad_tracker is None:
            raise ValueError(
                "Gradient tracker is required for confidence maps, "
                "but was not provided."
            )
        pred_tracker = pred_tracker / grad_tracker

    pred_tracker_np = postprocess_preds(
        pred_tracker,
        export_confidence=export_confidence,
        softmax_output=softmax_output,
        class_dim=0,
    ).numpy(force=True)

    if apply_no_data_mask:
        pred_tracker_np, nodata_mask = mask_prediction(
            input_array, pred_tracker_np, no_data_value
        )
    else:
        nodata_mask = None

    if export_to_disk:
        export_profile = build_export_profile(profile, pred_tracker_np)
        # if executer has been passed, submit the save_prediction function to it,
        # to avoid blocking the main thread
        if save_executor:
            save_future = save_executor.submit(
                save_prediction,
                output_path,
                export_profile,
                pred_tracker_np,
                nodata_mask,
            )
            if save_futures is not None:
                save_futures.append(save_future)
        # otherwise save the prediction directly

        else:
            save_prediction(
                output_path=output_path,
                export_profile=export_profile,
                pred_tracker_np=pred_tracker_np,
                nodata_mask=nodata_mask,
            )

    if pbar:
        pbar.update(1)
    return pred_tracker_np


def collect_models(
    custom_models: Optional[Union[list[torch.nn.Module], torch.nn.Module]],
    inference_device: torch.device,
    inference_dtype: torch.dtype,
    source: str,
    destination_model_dir: Union[str, Path, None] = None,
    model_version: float | None = None,
    compile_models: bool = False,
    patch_size: int | tuple[int, int] = 1000,
    batch_size: int = 1,
    compile_mode: str = "default",
) -> list[torch.nn.Module]:
    if custom_models is None:
        models = []
        for model_details in get_models(
            model_dir=destination_model_dir, source=source, model_version=model_version
        ):
            models.append(
                load_model_from_weights(
                    model_name=model_details["timm_model_name"],
                    model_library=model_details["model_library"],
                    weights_path=model_details["Path"],
                    device=inference_device,
                    dtype=inference_dtype,
                    compile_models=compile_models,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    compile_mode=compile_mode,
                )
            )
    else:
        # if not a list, make it a list of models
        if not isinstance(custom_models, list):
            custom_models = [custom_models]

        models = [
            model.to(device=inference_device, dtype=inference_dtype)
            for model in custom_models
        ]

        if compile_models:
            models = [
                compile_torch_model(
                    model,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    dtype=inference_dtype,
                    device=inference_device,
                    compile_mode=compile_mode,
                )
                for model in models
            ]

    # Patch models for MPS if needed
    if inference_device.type == "mps":
        models = patch_models_for_mps(
            models=models,
            inference_device=inference_device,
            inference_dtype=inference_dtype,
        )

    return models


def predict_from_array(
    input_array: np.ndarray,
    patch_size: int = 1000,
    patch_overlap: int = 300,
    batch_size: int = 1,
    inference_device: Optional[Union[str, torch.device]] = None,
    mosaic_device: Optional[Union[str, torch.device]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    export_confidence: bool = False,
    softmax_output: bool = True,
    no_data_value: int | float = 0,
    apply_no_data_mask: bool = True,
    custom_models: Optional[Union[list[torch.nn.Module], torch.nn.Module]] = None,
    pred_classes: int = 4,
    destination_model_dir: Union[str, Path, None] = None,
    model_download_source: str = "hugging_face",
    compile_models: bool = False,
    compile_mode: str = "default",
    model_version: float | None = None,
) -> np.ndarray:
    """Predict a cloud and cloud shadow mask from a Red, Green and NIR numpy array, with a spatial res between 10 m and 50 m.

    Args:
        input_array (np.ndarray): A numpy array with shape (3, height, width) representing the Red, Green and NIR bands.
        patch_size (int, optional): Size of the patches for inference. Defaults to 1000.
        patch_overlap (int, optional): Overlap between patches for inference. Defaults to 300.
        batch_size (int, optional): Number of patches to process in a batch. Defaults to 1.
        inference_device (Union[str, torch.device], optional): Device to use for inference (e.g., 'cpu', 'cuda', 'mps'). Defaults to None then default_device().
        mosaic_device (Union[str, torch.device], optional): Device to use for mosaicking patches. Defaults to inference device.
        inference_dtype (Union[torch.dtype, str], optional): Data type for inference. Defaults to torch.float32.
        export_confidence (bool, optional): If True, exports confidence maps instead of predicted classes. Defaults to False.
        softmax_output (bool, optional): If True, applies a softmax to the output, only used if export_confidence = True. Defaults to True.
        no_data_value (int, optional): Value within input scenes that specifies no data region. Defaults to 0.
        apply_no_data_mask (bool, optional): If True, applies a no-data mask to the predictions. Defaults to True.
        custom_models Union[list[torch.nn.Module], torch.nn.Module], optional): A list or singular custom torch models to use for prediction. Defaults to None.
        pred_classes (int, optional): Number of classes to predict. Defaults to 4, to be used with custom models. Defaults to 4.
        destination_model_dir Union[str, Path, None]: Directory to save the model weights. Defaults to None.
        model_download_source (str, optional): Source from which to download the model weights. Defaults to "hugging_face", can also be "google_drive".
        compile_models (bool, optional): If True, compiles the models for faster inference. Defaults to False.
        compile_mode (str, optional): Compilation mode for the models. Defaults to "default".
        model_version (float, optional): Version of the model to use. Defaults to the latest available version. Can also be set to 4.0, 3.0, 2.0, or 1.0 for older models.
    Returns:
        np.ndarray: A numpy array with shape (1, height, width) or (4, height, width if export_confidence = True) representing the predicted cloud and cloud shadow mask.

    """  # noqa: E501

    if inference_device is None:
        inference_device = default_device()

    inference_device = torch.device(inference_device)
    if mosaic_device is None:
        mosaic_device = inference_device
    else:
        mosaic_device = torch.device(mosaic_device)

    inference_dtype = get_torch_dtype(inference_dtype)

    # Warn if using CPU with reduced precision
    warn_cpu_reduced_precision(inference_device, inference_dtype)

    with device_lock(inference_device, mosaic_device):
        # if no custom model paths are provided, use the default models
        models = collect_models(
            custom_models=custom_models,
            inference_device=inference_device,
            inference_dtype=inference_dtype,
            source=model_download_source,
            destination_model_dir=destination_model_dir,
            model_version=model_version,
            compile_models=compile_models,
            patch_size=patch_size,
            batch_size=batch_size,
            compile_mode=compile_mode,
        )

        pred_tracker = coordinator(
            input_array=input_array,
            models=models,
            inference_device=inference_device,
            mosaic_device=mosaic_device,
            inference_dtype=inference_dtype,
            export_confidence=export_confidence,
            softmax_output=softmax_output,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            no_data_value=no_data_value,
            export_to_disk=False,
            apply_no_data_mask=apply_no_data_mask,
            pred_classes=pred_classes,
        )

    return pred_tracker


def predict_from_load_func(
    scene_paths: Union[list[Path], list[str]],
    load_func: Callable,
    patch_size: int = 1000,
    patch_overlap: int = 300,
    batch_size: int = 1,
    inference_device: Optional[Union[str, torch.device]] = None,
    mosaic_device: Optional[Union[str, torch.device]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    export_confidence: bool = False,
    softmax_output: bool = True,
    no_data_value: int | float = 0,
    overwrite: bool = True,
    apply_no_data_mask: bool = True,
    output_dir: Optional[Union[Path, str]] = None,
    custom_models: Optional[Union[list[torch.nn.Module], torch.nn.Module]] = None,
    pred_classes: int = 4,
    destination_model_dir: Union[str, Path, None] = None,
    model_download_source: str = "hugging_face",
    compile_models: bool = False,
    compile_mode: str = "default",
    model_version: float | None = None,
) -> list[Path]:
    """
    Predicts cloud and cloud shadow masks for a list of scenes using a specified loading function.

    Args:
        scene_paths (Union[list[Path], list[str]]): A list of paths to the scene files to be processed.
        load_func (Callable): A function to load the scene data. This function should take an input_path parameter and return a R,G,NIR numpy array and a rasterio for export profile, several load func are provided within data_loaders.py
        patch_size (int, optional): Size of the patches for inference. Defaults to 1000.
        patch_overlap (int, optional): Overlap between patches for inference. Defaults to 300.
        batch_size (int, optional): Number of patches to process in a batch. Defaults to 1.
        inference_device (Union[str, torch.device], optional): Device to use for inference (e.g., 'cpu', 'cuda', 'mps'). Defaults to None then default_device().
        mosaic_device (Union[str, torch.device], optional): Device to use for mosaicking patches. Defaults to inference device.
        inference_dtype (Union[torch.dtype, str], optional): Data type for inference. Defaults to torch.float32.
        export_confidence (bool, optional): If True, exports confidence maps instead of predicted classes. Defaults to False.
        softmax_output (bool, optional): If True, applies a softmax to the output, only used if export_confidence = True. Defaults to True.
        no_data_value (int, optional): Value within input scenes that specifies no data region. Defaults to 0.
        overwrite (bool, optional): If False, skips scenes that already have a prediction file. Defaults to True.
        apply_no_data_mask (bool, optional): If True, applies a no-data mask to the predictions. Defaults to True.
        output_dir (Optional[Union[Path, str]], optional): Directory to save the prediction files. Defaults to None. If None, the predictions will be saved in the same directory as the input scene.
        custom_models Union[list[torch.nn.Module], torch.nn.Module], optional): A list or singular custom torch models to use for prediction. Defaults to None.
        pred_classes (int, optional): Number of classes to predict. Defaults to 4, to be used with custom models. Defaults to 4.
        destination_model_dir Union[str, Path, None]: Directory to save the model weights. Defaults to None.
        model_download_source (str, optional): Source from which to download the model weights. Defaults to "hugging_face", can also be "google_drive".
        compile_models (bool, optional): If True, compiles the models for faster inference. Defaults to False.
        compile_mode (str, optional): Compilation mode for the models. Defaults to "default".
        model_version (float, optional): Version of the model to use. Defaults to the latest available version. Can also be set to 4.0, 3.0, 2.0, or 1.0 for older models.
    Returns:
        list[Path]: A list of paths to the output prediction files.

    """  # noqa: E501
    if inference_device is None:
        inference_device = default_device()
    pred_paths = []

    inference_device = torch.device(inference_device)
    if mosaic_device is None:
        mosaic_device = inference_device
    else:
        mosaic_device = torch.device(mosaic_device)

    inference_dtype = get_torch_dtype(inference_dtype)

    # Warn if using CPU with reduced precision
    warn_cpu_reduced_precision(inference_device, inference_dtype)

    with device_lock(inference_device, mosaic_device):
        models = collect_models(
            custom_models=custom_models,
            inference_device=inference_device,
            inference_dtype=inference_dtype,
            destination_model_dir=destination_model_dir,
            source=model_download_source,
            model_version=model_version,
            compile_models=compile_models,
            patch_size=patch_size,
            batch_size=batch_size,
            compile_mode=compile_mode,
        )

    pbar = tqdm(
        total=len(scene_paths),
        desc=f"Running inference using {inference_device.type} "
        f"{str(inference_dtype).split('.')[-1]}",
    )

    # Inference runs in a background worker so the next scene loads in parallel,
    # future.result() re-raises any error from the worker in this thread
    inf_executor = ThreadPoolExecutor(max_workers=1)
    save_executor = ThreadPoolExecutor(max_workers=1)
    inf_future: Optional[Future] = None
    save_futures: list[Future] = []

    try:
        for scene_path in scene_paths:
            scene_path = Path(scene_path)
            output_path = make_output_path(scene_path, output_dir)

            pred_paths.append(output_path)

            if output_path.exists() and not overwrite:
                pbar.update(1)
                pbar.refresh()
                continue

            input_array, profile = load_func(input_path=scene_path)

            # wait for the previous scene before starting the next
            if inf_future is not None:
                inf_future.result()

            inf_future = inf_executor.submit(
                locked_coordinator,
                input_array=input_array,
                profile=profile,
                output_path=output_path,
                models=models,
                inference_dtype=inference_dtype,
                export_confidence=export_confidence,
                softmax_output=softmax_output,
                inference_device=inference_device,
                mosaic_device=mosaic_device,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                batch_size=batch_size,
                no_data_value=no_data_value,
                pbar=pbar,
                apply_no_data_mask=apply_no_data_mask,
                save_executor=save_executor,
                pred_classes=pred_classes,
                save_futures=save_futures,
            )

        if inf_future is not None:
            inf_future.result()
    finally:
        inf_executor.shutdown(wait=True)
        save_executor.shutdown(wait=True)
        pbar.refresh()
        if inference_device.type.startswith("cuda"):
            torch.cuda.empty_cache()

    # surface any errors raised while saving
    for future in save_futures:
        future.result()

    return pred_paths


# marks the end of an iterator
_SENTINEL = object()


def _prepare_item(
    item: Any, load_func: Optional[Callable]
) -> tuple[Union[np.ndarray, torch.Tensor], Optional[Profile]]:
    """Load an item with load_func if provided and cast numpy images to float32.
    Tensors are returned as they are and cast when collated, since casting a tensor
    on MPS is MPS work that must hold the device lock. Returns the image and the
    export profile, if load_func returned one."""
    profile = None
    if load_func is not None:
        item = load_func(input_path=item)
        if isinstance(item, tuple):
            item, profile = item[0], item[1]

    if isinstance(item, torch.Tensor):
        return item, profile
    return np.asarray(item, dtype=np.float32), profile


def _check_item_shape(
    image: Union[np.ndarray, torch.Tensor], index: int, expected_shape: tuple
) -> None:
    """Check an image matches the shape of the first image."""
    if tuple(image.shape) != expected_shape:
        raise ValueError(
            f"All images must have the same shape. Image at index {index} has shape "
            f"{tuple(image.shape)}, expected {expected_shape} (shape of the first "
            f"image). Call predict_from_batch separately for each image size."
        )


def _iter_batches(
    first: tuple[Union[np.ndarray, torch.Tensor], Optional[Profile]],
    first_item: Any,
    items: Iterator[Any],
    load_func: Optional[Callable],
    batch_size: int,
    expected_shape: tuple,
) -> Generator[tuple[list, list[Optional[Profile]], list[Any]], None, None]:
    """Yield batches of (images, profiles, original items) in input order.
    Items are loaded ahead in a thread pool so loading overlaps with inference."""
    num_workers = max(1, min(batch_size, os.cpu_count() or 1))
    max_pending = batch_size * 2

    images, profiles, batch_items = [first[0]], [first[1]], [first_item]
    index = 1

    executor = ThreadPoolExecutor(max_workers=num_workers)
    pending: deque[tuple[Any, Future]] = deque()

    def submit_next() -> None:
        item = next(items, _SENTINEL)
        if item is not _SENTINEL:
            pending.append((item, executor.submit(_prepare_item, item, load_func)))

    try:
        for _ in range(max_pending):
            submit_next()

        while pending:
            if len(images) == batch_size:
                yield images, profiles, batch_items
                images, profiles, batch_items = [], [], []

            item, future = pending.popleft()
            image, profile = future.result()
            submit_next()

            _check_item_shape(image, index, expected_shape)
            index += 1

            images.append(image)
            profiles.append(profile)
            batch_items.append(item)

        yield images, profiles, batch_items
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _collate(images: list, device: torch.device) -> torch.Tensor:
    """Stack images into a float32 (B, C, H, W) tensor, pinned when numpy images
    will be copied to a CUDA device. Tensor images are stacked on the device."""
    if all(isinstance(image, np.ndarray) for image in images):
        tensor = torch.from_numpy(np.stack(images))
        return tensor.pin_memory() if device.type == "cuda" else tensor
    # Move, then cast. Doing both in one .to() call returns wrong values for fp16
    # MPS views copied to the CPU (PyTorch 2.11)
    stacked = torch.stack(
        [torch.as_tensor(image).to(device=device) for image in images]
    )
    return stacked.to(dtype=torch.float32)


def _batch_devices(batch: Union[list, torch.Tensor]) -> list[torch.device]:
    """Devices of the tensors in a batch, collated or not."""
    if isinstance(batch, torch.Tensor):
        return [batch.device]
    return [image.device for image in batch if isinstance(image, torch.Tensor)]


def _collate_in_background(images: list, device: torch.device) -> bool:
    """Whether images can be collated on the prefetch thread. Collation that touches
    MPS, moving tensors to or from it or casting them on it, is left to the caller,
    which holds the device lock, since PyTorch's MPS backend isn't thread-safe."""
    has_tensors = any(isinstance(image, torch.Tensor) for image in images)
    touches_mps = (device.type == "mps" and has_tensors) or any(
        d.type == "mps" for d in _batch_devices(images)
    )
    return not touches_mps


def _prefetch(
    batches: Generator[tuple[list, list[Optional[Profile]], list[Any]], None, None],
    device: torch.device,
    depth: int = 2,
) -> Generator[
    tuple[Union[torch.Tensor, list], list[Optional[Profile]], list[Any]], None, None
]:
    """Collate batches in a background thread so the next batch is ready while the
    current one runs on the device. Batches whose collation touches MPS are yielded
    as lists for the caller to collate under the device lock. Errors are re-raised
    in the caller, and closing this generator stops the thread."""
    result_queue: Queue = Queue(maxsize=depth)
    stop = Event()

    def put(item: Any) -> bool:
        # time out regularly so the producer can't block forever on a full queue
        while not stop.is_set():
            try:
                result_queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def producer() -> None:
        try:
            for images, profiles, items in batches:
                if _collate_in_background(images, device):
                    images = _collate(images, device)
                if not put((images, profiles, items)):
                    return
            put(_SENTINEL)
        except BaseException as e:
            put(e)
        finally:
            batches.close()

    thread = Thread(target=producer, daemon=True)
    thread.start()
    try:
        while True:
            result = result_queue.get()
            if result is _SENTINEL:
                return
            if isinstance(result, BaseException):
                raise result
            yield result
    finally:
        stop.set()
        thread.join()


def _predict_batch(
    batch: Union[torch.Tensor, list],
    models: list[torch.nn.Module],
    inference_device: torch.device,
    inference_dtype: torch.dtype,
    no_data_value: int | float,
    export_confidence: bool,
    softmax_output: bool,
    apply_no_data_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a batch and return the predictions and the valid data masks as numpy
    arrays. All device work happens here so the caller can hold the device lock
    around it, and device tensors are freed before it's released."""
    if not isinstance(batch, torch.Tensor):
        batch = _collate(batch, inference_device)
    raw_batch = batch.to(device=inference_device, non_blocking=True)
    valid_mask = batch_nodata_mask(raw_batch, no_data_value)
    norm_batch = channel_norm_torch(raw_batch, no_data_value).to(dtype=inference_dtype)

    preds = postprocess_preds(
        infer_batch(models, norm_batch),
        export_confidence=export_confidence,
        softmax_output=softmax_output,
        class_dim=1,
    )

    # match the tiled path, which skips patches that are all no data
    empty = ~valid_mask.flatten(start_dim=1).any(dim=1)
    preds = preds.masked_fill(
        empty[:, None, None, None],
        np.nan if export_confidence and not softmax_output else 0,
    )
    if apply_no_data_mask:
        preds = preds * valid_mask

    return (
        preds.numpy(force=True),
        valid_mask[:, 0].to(dtype=torch.uint8).numpy(force=True),
    )


# The return type depends on export_to_disk: an array of predictions, or the
# paths of the exported files. Overloads let type checkers infer which.
@overload
def predict_from_batch(
    data: Iterable[Any],
    batch_size: int = 1,
    load_func: Optional[Callable] = None,
    inference_device: Optional[Union[str, torch.device]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    export_confidence: bool = False,
    softmax_output: bool = True,
    no_data_value: int | float = 0,
    apply_no_data_mask: bool = True,
    export_to_disk: Literal[False] = False,
    output_dir: Optional[Union[Path, str]] = None,
    overwrite: bool = True,
    custom_models: Optional[Union[list[torch.nn.Module], torch.nn.Module]] = None,
    destination_model_dir: Union[str, Path, None] = None,
    model_download_source: str = "hugging_face",
    compile_models: bool = False,
    compile_mode: str = "default",
    model_version: float | None = None,
) -> np.ndarray: ...


@overload
def predict_from_batch(
    data: Iterable[Any],
    batch_size: int = 1,
    load_func: Optional[Callable] = None,
    inference_device: Optional[Union[str, torch.device]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    export_confidence: bool = False,
    softmax_output: bool = True,
    no_data_value: int | float = 0,
    apply_no_data_mask: bool = True,
    *,
    export_to_disk: Literal[True],
    output_dir: Optional[Union[Path, str]] = None,
    overwrite: bool = True,
    custom_models: Optional[Union[list[torch.nn.Module], torch.nn.Module]] = None,
    destination_model_dir: Union[str, Path, None] = None,
    model_download_source: str = "hugging_face",
    compile_models: bool = False,
    compile_mode: str = "default",
    model_version: float | None = None,
) -> list[Path]: ...


@overload
def predict_from_batch(
    data: Iterable[Any],
    batch_size: int = 1,
    load_func: Optional[Callable] = None,
    inference_device: Optional[Union[str, torch.device]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    export_confidence: bool = False,
    softmax_output: bool = True,
    no_data_value: int | float = 0,
    apply_no_data_mask: bool = True,
    export_to_disk: bool = False,
    output_dir: Optional[Union[Path, str]] = None,
    overwrite: bool = True,
    custom_models: Optional[Union[list[torch.nn.Module], torch.nn.Module]] = None,
    destination_model_dir: Union[str, Path, None] = None,
    model_download_source: str = "hugging_face",
    compile_models: bool = False,
    compile_mode: str = "default",
    model_version: float | None = None,
) -> Union[np.ndarray, list[Path]]: ...


def predict_from_batch(
    data: Iterable[Any],
    batch_size: int = 1,
    load_func: Optional[Callable] = None,
    inference_device: Optional[Union[str, torch.device]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    export_confidence: bool = False,
    softmax_output: bool = True,
    no_data_value: int | float = 0,
    apply_no_data_mask: bool = True,
    export_to_disk: bool = False,
    output_dir: Optional[Union[Path, str]] = None,
    overwrite: bool = True,
    custom_models: Optional[Union[list[torch.nn.Module], torch.nn.Module]] = None,
    destination_model_dir: Union[str, Path, None] = None,
    model_download_source: str = "hugging_face",
    compile_models: bool = False,
    compile_mode: str = "default",
    model_version: float | None = None,
) -> Union[np.ndarray, list[Path]]:
    """Predict cloud and cloud shadow masks for many same-sized images, such as the chips of an existing dataset, running several images per batch.

    Each image is predicted as a single patch, so there is no patch overlap or mosaicking. All images in a call must have the same shape.

    Args:
        data (Iterable): An iterable of (3, height, width) Red, Green and NIR numpy arrays or torch tensors, e.g. a 4D array or tensor, a list of arrays or a generator. If load_func is provided, an iterable of items (usually file paths) to pass to load_func.
        batch_size (int, optional): Number of images to process in a batch. Defaults to 1.
        load_func (Callable, optional): A function called as load_func(input_path=item) that returns a (3, height, width) array, or an (array, rasterio profile) tuple such as the loaders in data_loaders.py. It runs on background threads, so it should return numpy arrays or CPU tensors; creating MPS tensors inside it isn't serialised with inference. Defaults to None.
        inference_device (Union[str, torch.device], optional): Device to use for inference (e.g., 'cpu', 'cuda', 'mps'). Defaults to None then default_device().
        inference_dtype (Union[torch.dtype, str], optional): Data type for inference. Defaults to torch.float32.
        export_confidence (bool, optional): If True, exports confidence maps instead of predicted classes. Defaults to False.
        softmax_output (bool, optional): If True, applies a softmax to the output, only used if export_confidence = True. Defaults to True.
        no_data_value (int, optional): Value within input images that specifies no data region. Defaults to 0.
        apply_no_data_mask (bool, optional): If True, applies a no-data mask to the predictions. Defaults to True.
        export_to_disk (bool, optional): If True, saves each prediction as a GeoTIFF and returns the file paths instead of an array. Requires load_func and path-like items. Defaults to False.
        output_dir (Optional[Union[Path, str]], optional): Directory to save the prediction files, only used if export_to_disk = True. Defaults to None, which saves next to each input file.
        overwrite (bool, optional): If False, skips items that already have a prediction file, only used if export_to_disk = True. Defaults to True.
        custom_models Union[list[torch.nn.Module], torch.nn.Module], optional): A list or singular custom torch models to use for prediction. Defaults to None.
        destination_model_dir Union[str, Path, None]: Directory to save the model weights. Defaults to None.
        model_download_source (str, optional): Source from which to download the model weights. Defaults to "hugging_face", can also be "google_drive".
        compile_models (bool, optional): If True, compiles the models for faster inference. Defaults to False.
        compile_mode (str, optional): Compilation mode for the models. Defaults to "default".
        model_version (float, optional): Version of the model to use. Defaults to the latest available version. Can also be set to 4.0, 3.0, 2.0, or 1.0 for older models.
    Returns:
        Union[np.ndarray, list[Path]]: A numpy array with shape (N, 1, height, width) of predicted classes, or (N, classes, height, width) if export_confidence = True. If export_to_disk = True, a list of N prediction file paths instead.

    """  # noqa: E501
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, found {batch_size}.")

    output_paths: list[Path] = []
    if export_to_disk:
        if load_func is None:
            raise ValueError(
                "export_to_disk requires load_func, as output file names and "
                "profiles come from the loaded files."
            )
        data = list(data)
        for item in data:
            if not isinstance(item, (str, os.PathLike)):
                raise ValueError(
                    "export_to_disk requires data to be file paths, "
                    f"found item of type {type(item).__name__}."
                )
        output_paths = [make_output_path(item, output_dir) for item in data]
        if not overwrite:
            data = [item for item, path in zip(data, output_paths) if not path.exists()]
            if not data:
                return output_paths

    try:
        total: Optional[int] = len(data)  # type: ignore[arg-type]
    except TypeError:
        total = None

    items = iter(data)
    first_item = next(items, _SENTINEL)
    if first_item is _SENTINEL:
        raise ValueError("data is empty, at least one image is required.")

    first = _prepare_item(first_item, load_func)
    image_shape = tuple(first[0].shape)
    validate_input_shape(image_shape)

    if inference_device is None:
        inference_device = default_device()
    inference_device = torch.device(inference_device)
    inference_dtype = get_torch_dtype(inference_dtype)

    # Warn if using CPU with reduced precision
    warn_cpu_reduced_precision(inference_device, inference_dtype)

    with device_lock(inference_device):
        models = collect_models(
            custom_models=custom_models,
            inference_device=inference_device,
            inference_dtype=inference_dtype,
            source=model_download_source,
            destination_model_dir=destination_model_dir,
            model_version=model_version,
            compile_models=compile_models,
            patch_size=(image_shape[1], image_shape[2]),
            batch_size=batch_size,
            compile_mode=compile_mode,
        )

    model_channels = _get_model_in_channels(models[0])
    if image_shape[0] != model_channels:
        raise ValueError(
            f"Images must have {model_channels} bands (Red, Green, NIR), found "
            f"shape {image_shape}. If a band is unavailable, fill it with zeros."
        )

    pbar = tqdm(
        total=total,
        desc=f"Running inference using {inference_device.type} "
        f"{str(inference_dtype).split('.')[-1]}",
    )
    save_executor = ThreadPoolExecutor(max_workers=1) if export_to_disk else None
    save_futures: list[Future] = []
    results: list[np.ndarray] = []
    output: Optional[np.ndarray] = None
    written = 0

    try:
        batches = _iter_batches(
            first=first,
            first_item=first_item,
            items=items,
            load_func=load_func,
            batch_size=batch_size,
            expected_shape=image_shape,
        )
        # closing() stops the background threads if an error is raised mid-loop
        with closing(_prefetch(batches, inference_device)) as prefetched:
            for batch, profiles, batch_items in prefetched:
                # Lock per batch rather than per call, so the lock isn't held while
                # waiting on loaders, which may run MPS predictions themselves
                with device_lock(inference_device, *_batch_devices(batch)):
                    batch_preds, nodata_masks = _predict_batch(
                        batch,
                        models,
                        inference_device=inference_device,
                        inference_dtype=inference_dtype,
                        no_data_value=no_data_value,
                        export_confidence=export_confidence,
                        softmax_output=softmax_output,
                        apply_no_data_mask=apply_no_data_mask,
                    )

                if save_executor is not None:
                    for pred, profile, item, nodata_mask in zip(
                        batch_preds, profiles, batch_items, nodata_masks
                    ):
                        save_futures.append(
                            save_executor.submit(
                                save_prediction,
                                make_output_path(item, output_dir),
                                build_export_profile(profile, pred),
                                pred,
                                nodata_mask if apply_no_data_mask else None,
                            )
                        )
                elif total is not None:
                    if output is None:
                        output = np.empty(
                            (total, *batch_preds.shape[1:]), dtype=batch_preds.dtype
                        )
                    output[written : written + len(batch_preds)] = batch_preds
                    written += len(batch_preds)
                else:
                    results.append(batch_preds)

                pbar.update(len(batch_preds))
    finally:
        if save_executor is not None:
            save_executor.shutdown(wait=True)
        pbar.close()
        if inference_device.type.startswith("cuda"):
            torch.cuda.empty_cache()

    # surface any errors raised while saving
    for future in save_futures:
        future.result()

    if export_to_disk:
        return output_paths
    if output is not None:
        return output
    return np.concatenate(results)
