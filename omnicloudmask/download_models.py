import csv
from importlib import resources
from pathlib import Path
from typing import Union

import gdown
import platformdirs
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .__version__ import __version__ as omnicloudmask_version

model_index_path = resources.files("omnicloudmask") / "model_download_links.csv"


def get_latest_model_version() -> float:
    """Get the latest model version from the model_download_links.csv file"""
    with model_index_path.open() as f:
        reader = csv.DictReader(f)
        models = list(reader)

    versions = [float(model["version"]) for model in models]
    return max(versions)


def download_file_from_google_drive(file_id: str, destination: Path) -> None:
    """
    Downloads a file from Google Drive and saves it at the given destination
    using gdown.

    Args:
        file_id (str): The ID of the file on Google Drive.
        destination (Path): The local path where the file should be saved.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(destination), quiet=False)


def download_file_from_hugging_face(destination: Path) -> None:
    """
    Downloads a file from Hugging Face and saves it at the given destination using
    hf_hub_download.
    Loads the resulting safetensors file and saves it as a PyTorch model state for
    compatibility with the rest of the codebase.

    Args:
        file_id (str): The ID of the file on Hugging Face.
        destination (Path): The local path where the file should be saved.
    """
    file_name = destination.stem
    safetensor_path = hf_hub_download(
        repo_id="NickWright/OmniCloudMask",
        filename=f"{file_name}.safetensors",
        force_download=True,
        cache_dir=destination.parent,
        local_dir=destination.parent,
    )

    # If using the v1 weights then the file will be saved as a .pth file
    if destination.suffix == ".pth":
        model_state = load_file(safetensor_path)
        torch.save(model_state, destination)


def download_file(file_id: str, destination: Path, source: str) -> None:
    if source == "google_drive":
        download_file_from_google_drive(file_id, destination)
    elif source == "hugging_face":
        download_file_from_hugging_face(destination)
    else:
        raise ValueError(
            "Invalid source. Supported sources are 'google_drive' and 'hugging_face'."
        )


def get_model_data_dir() -> Path:
    """Get the user data directory for model files"""
    data_dir = Path(
        platformdirs.user_data_dir(
            "omnicloudmask", version=omnicloudmask_version, ensure_exists=True
        )
    )
    return data_dir


def get_models(
    force_download: bool = False,
    model_dir: Union[str, Path, None] = None,
    source: str = "hugging_face",
    model_version: float | None = None,
) -> list[dict]:
    """
    Downloads the model weights from Google Drive and saves them locally.

    Args:
        force_download (bool): Whether to force download the model weights even if they
        already exist locally.
        model_dir (Union[str, Path, None]): The directory where the model weights
        should be saved.
        source (str): The source from which the model weights should be downloaded.
        Currently, only "google_drive" or "hugging_face" are supported.
        model_version (float | None): Version of the model to use. Defaults to
        the latest available version if None. Can be set to 4.0, 3.0, 2.0, or 1.0.
    """

    with model_index_path.open() as f:
        reader = csv.DictReader(f)
        models = list(reader)

    if model_dir is not None:
        model_dir = Path(model_dir)
    else:
        model_dir = get_model_data_dir()

    if model_version is None:
        model_version = get_latest_model_version()

    available_versions = []
    model_paths = []

    for model_dict in models:
        current_model_version = float(model_dict["version"])
        available_versions.append(current_model_version)
        if model_version != current_model_version:
            continue

        model_dir.mkdir(exist_ok=True)
        destination = model_dir / str(model_dict["file_name"])

        if (
            not destination.exists()
            or force_download
            or destination.stat().st_size <= 1024 * 1024
        ):
            download_file(
                file_id=str(model_dict["google_drive_id"]),
                destination=destination,
                source=source,
            )

        model_paths.append(
            {
                "Path": destination,
                "timm_model_name": model_dict["timm_model_name"],
                "model_library": model_dict["model_library"],
            }
        )

    if not model_paths:
        raise ValueError(
            f"""Model version {model_version} not found. 
            Available versions: {sorted(set(available_versions))}"""
        )

    return model_paths
