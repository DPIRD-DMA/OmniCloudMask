import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Thread
from typing import Callable, Generator, Optional, Union

import numpy as np
import torch
from rasterio.profiles import Profile
from tqdm.auto import tqdm

from .__version__ import __version__
from .download_models import get_models
from .model_utils import (
    create_gradient_mask,
    default_device,
    get_torch_dtype,
    inference_and_store,
    load_model,
)
from .raster_utils import (
    get_patch,
    make_patch_indexes,
    mask_prediction,
    save_prediction,
)


def compile_batches(
    batch_size: int,
    patch_size: int,
    patch_indexes: list[tuple[int, int, int, int]],
    input_array: np.ndarray,
    no_data_value: int,
    inference_device: torch.device,
    inference_dtype: torch.dtype,
) -> Generator[tuple[torch.Tensor, list[tuple[int, int, int, int]]], None, None]:
    """Used to compile batches of patches from the input array and return them as a generator."""

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(get_patch, input_array, index, no_data_value)
            for index in patch_indexes
        ]

        total_futures = len(futures)
        all_indexes = set()
        index_batch = []
        patch_batch_array = np.zeros(
            (batch_size, input_array.shape[0], patch_size, patch_size), dtype=np.float32
        )

        for index, future in enumerate(as_completed(futures)):
            patch, new_index = future.result()

            if patch is not None and new_index not in all_indexes:
                index_batch.append(new_index)
                patch_batch_array[len(index_batch) - 1] = patch
                all_indexes.add(new_index)

            if len(index_batch) == batch_size or index == total_futures - 1:
                if len(index_batch) == 0:
                    continue
                input_tensor = (
                    torch.tensor(patch_batch_array[: len(index_batch)])
                    .to(inference_device)
                    .to(inference_dtype)
                )
                yield input_tensor, index_batch
                index_batch = []


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
    no_data_value: int = 0,
) -> None:
    """Used to execute the model on the input array, in patches. Predictions are stored in pred_tracker and grad_tracker, updated in place."""
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
    inf_thread = Thread()
    for patch_batch, index_batch in input_tensor_gen:
        if inf_thread.is_alive():
            inf_thread.join()

        inf_thread = Thread(
            target=inference_and_store,
            kwargs={
                "models": models,
                "patch_batch": patch_batch,
                "index_batch": index_batch,
                "pred_tracker": pred_tracker,
                "gradient": gradient,
                "grad_tracker": grad_tracker,
            },
        )
        inf_thread.start()

    if inf_thread.is_alive():
        inf_thread.join()


def check_patch_size(
    input_array: np.ndarray, no_data_value: int, patch_size: int, patch_overlap: int
) -> tuple[int, int]:
    """Used to check the inputs and adjust the patch size and overlap if necessary."""

    # if the input has a lot of no data values and the patch size is larger than half the image size, we reduce the patch size and overlap
    if np.count_nonzero(input_array == no_data_value) / input_array.size > 0.3:
        if patch_size > min(input_array.shape[1], input_array.shape[2]) / 2:
            patch_size = min(input_array.shape[1], input_array.shape[2]) // 2
            if patch_size // 2 < patch_overlap:
                patch_overlap = patch_size // 2

            warnings.warn(
                f"Significant no-data areas detected. Adjusting patch size to {patch_size}px and overlap to {patch_overlap}px to minimize no-data patches."
            )
    # if the patch size is larger than the image size, we reduce the patch size and overlap
    if patch_size > min(input_array.shape[1], input_array.shape[2]):
        patch_size = min(input_array.shape[1], input_array.shape[2])
        if patch_size // 2 < patch_overlap:
            patch_overlap = patch_size // 2
        warnings.warn(
            f"Patch size too large, reducing to {patch_size} and overlap to {patch_overlap}."
        )

    # if the patch overlap is larger than the patch size, raise an error
    if patch_overlap >= patch_size:
        raise ValueError(
            f"Patch overlap {patch_overlap}px must be less than patch size {patch_size}px."
        )
    return patch_overlap, patch_size


def coordinator(
    input_array: np.ndarray,
    models: list[torch.nn.Module],
    inference_dtype: torch.dtype,
    export_confidence: bool,
    inference_device: torch.device,
    mosaic_device: torch.device,
    patch_size: int,
    patch_overlap: int,
    batch_size: int,
    profile: Profile = Profile(),
    output_path: Path = Path(""),
    no_data_value: int = 0,
    pbar: Optional[tqdm] = None,
    apply_no_data_mask: bool = False,
    export_to_disk: bool = True,
    save_executor: Optional[ThreadPoolExecutor] = None,
    pred_classes: int = 4,
) -> np.ndarray:
    """Used to coordinate the process of predicting from an input array."""

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
        pred_tracker = torch.clip(
            (torch.nn.functional.softmax((pred_tracker / grad_tracker), 0) + 0.001),
            0.001,
            0.999,
        )

        pred_tracker_np = pred_tracker.float().numpy(force=True)

    else:
        pred_tracker_np = (
            torch.argmax(pred_tracker, 0, keepdim=True)
            .numpy(force=True)
            .astype(np.uint8)
        )

    if apply_no_data_mask:
        pred_tracker = mask_prediction(input_array, pred_tracker_np, no_data_value)

    if export_to_disk:
        export_profile = profile.copy()
        export_profile.update(
            dtype=pred_tracker.dtype,
            count=pred_tracker.shape[0],
            compress="lzw",
            nodata=0,
            driver="GTiff",
        )
        # if executer has been passed, submit the save_prediction function to it, to avoid blocking the main thread
        if save_executor:
            save_executor.submit(
                save_prediction, output_path, export_profile, pred_tracker_np
            )
        # otherwise save the prediction directly

        else:
            save_prediction(output_path, export_profile, pred_tracker_np)

    if pbar:
        pbar.update(1)
    return pred_tracker_np


def predict_from_array(
    input_array: np.ndarray,
    patch_size: int = 1000,
    patch_overlap: int = 300,
    batch_size: int = 1,
    inference_device: Union[str, torch.device] = default_device(),
    mosaic_device: Optional[Union[str, torch.device]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    export_confidence: bool = False,
    no_data_value: int = 0,
    apply_no_data_mask: bool = True,
) -> np.ndarray:
    """Predict a cloud and cloud shadow mask from a Red, Green and NIR numpy array, with a spatial res between 10 m and 50 m.

    Args:
        input_array (np.ndarray): A numpy array with shape (3, height, width) representing the Red, Green and NIR bands.
        patch_size (int, optional): Size of the patches for inference. Defaults to 1000.
        patch_overlap (int, optional): Overlap between patches for inference. Defaults to 300.
        batch_size (int, optional): Number of patches to process in a batch. Defaults to 1.
        inference_device (Union[str, torch.device], optional): Device to use for inference (e.g., 'cpu', 'cuda', 'mps'). Defaults to the device returned by default_device().
        mosaic_device (Union[str, torch.device], optional): Device to use for mosaicking patches. Defaults to inference device.
        inference_dtype (Union[torch.dtype, str], optional): Data type for inference. Defaults to torch.float32.
        export_confidence (bool, optional): If True, exports confidence maps instead of with predicted classes. Defaults to False.
        no_data_value (int, optional): Value within input scenes that specifies no data region. Defaults to 0.
        apply_no_data_mask (bool, optional): If True, applies a no-data mask to the predictions. Defaults to True.

    Returns:
        np.ndarray: A numpy array with shape (1, height, width) or (4, height, width if export_confidence = True) representing the predicted cloud and cloud shadow mask.

    """

    inference_device = torch.device(inference_device)
    if mosaic_device is None:
        mosaic_device = inference_device
    else:
        mosaic_device = torch.device(mosaic_device)

    inference_dtype = get_torch_dtype(inference_dtype)

    models = [
        load_model(model_path, inference_device, inference_dtype)
        for model_path in get_models()
    ]

    pred_tracker = coordinator(
        input_array=input_array,
        models=models,
        inference_device=inference_device,
        mosaic_device=mosaic_device,
        inference_dtype=inference_dtype,
        export_confidence=export_confidence,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        no_data_value=no_data_value,
        export_to_disk=False,
        apply_no_data_mask=apply_no_data_mask,
    )

    return pred_tracker


def predict_from_load_func(
    scene_paths: Union[list[Path], list[str]],
    load_func: Callable,
    patch_size: int = 1000,
    patch_overlap: int = 300,
    batch_size: int = 1,
    inference_device: Union[str, torch.device] = default_device(),
    mosaic_device: Optional[Union[str, torch.device]] = None,
    inference_dtype: Union[torch.dtype, str] = torch.float32,
    export_confidence: bool = False,
    no_data_value: int = 0,
    overwrite: bool = True,
    apply_no_data_mask: bool = True,
) -> list[Path]:
    """
    Predicts cloud and cloud shadow masks for a list of scenes using a specified loading function.

    Args:
        scene_paths (Union[list[Path], list[str]]): A list of paths to the scene files to be processed.
        load_func (Callable): A function to load the scene data. This function should take an input_path parameter and return a R,G,NIR numpy array and a rasterio for export profile, several load func are provided within data_loaders.py
        patch_size (int, optional): Size of the patches for inference. Defaults to 1000.
        patch_overlap (int, optional): Overlap between patches for inference. Defaults to 300.
        batch_size (int, optional): Number of patches to process in a batch. Defaults to 1.
        inference_device (Union[str, torch.device], optional): Device to use for inference (e.g., 'cpu', 'cuda', 'mps'). Defaults to the device returned by default_device().
        mosaic_device (Union[str, torch.device], optional): Device to use for mosaicking patches. Defaults to inference device.
        inference_dtype (Union[torch.dtype, str], optional): Data type for inference. Defaults to torch.float32.
        export_confidence (bool, optional): If True, exports confidence maps instead of with predicted classes. Defaults to False.
        no_data_value (int, optional): Value within input scenes that specifies no data region. Defaults to 0.
        overwrite (bool, optional): If False, skips scenes that already have a prediction file. Defaults to True.
        apply_no_data_mask (bool, optional): If True, applies a no-data mask to the predictions. Defaults to True.

    Returns:
        list[Path]: A list of paths to the output prediction files.

    """
    pred_paths = []
    inf_thread = Thread()
    save_executor = ThreadPoolExecutor(max_workers=1)

    inference_device = torch.device(inference_device)
    if mosaic_device is None:
        mosaic_device = inference_device
    else:
        mosaic_device = torch.device(mosaic_device)

    inference_dtype = get_torch_dtype(inference_dtype)

    models = [
        load_model(model_path, inference_device, inference_dtype)
        for model_path in get_models()
    ]

    pbar = tqdm(
        total=len(scene_paths),
        desc=f"Running inference using {inference_device.type} {str(inference_dtype).split('.')[-1]}",
    )

    for scene_path in scene_paths:
        scene_path = Path(scene_path)
        file_name = f"{scene_path.stem}_OCM_v{__version__.replace('.','_')}.tif"
        output_path = scene_path.parent / file_name
        pred_paths.append(output_path)

        if output_path.exists() and not overwrite:
            pbar.update(1)
            pbar.refresh()
            continue

        input_array, profile = load_func(input_path=scene_path)

        while inf_thread.is_alive():
            inf_thread.join()

        inf_thread = Thread(
            target=coordinator,
            kwargs={
                "input_array": input_array,
                "profile": profile,
                "output_path": output_path,
                "models": models,
                "inference_dtype": inference_dtype,
                "export_confidence": export_confidence,
                "inference_device": inference_device,
                "mosaic_device": mosaic_device,
                "patch_size": patch_size,
                "patch_overlap": patch_overlap,
                "batch_size": batch_size,
                "no_data_value": no_data_value,
                "pbar": pbar,
                "apply_no_data_mask": apply_no_data_mask,
                "save_executor": save_executor,
            },
        )
        inf_thread.start()

    while inf_thread.is_alive():
        inf_thread.join()

    if inference_device.type.startswith("cuda"):
        torch.cuda.empty_cache()

    save_executor.shutdown(wait=True)
    pbar.refresh()

    return pred_paths
