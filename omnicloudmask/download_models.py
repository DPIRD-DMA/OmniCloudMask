from importlib import resources
from pathlib import Path
from typing import Union

import gdown
import pandas as pd
import platformdirs
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .__version__ import __version__ as omnicloudmask_version


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
    model_version: float = 3.0,
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
    """

    with (resources.files("omnicloudmask") / "model_download_links.csv").open() as f:
        model_df = pd.read_csv(f)

    # set version column to float
    model_df["version"] = model_df["version"].astype(float)
    available_versions = model_df["version"].unique()
    if model_version not in available_versions:
        raise ValueError(
            f"""Model version {model_version} not found. 
            Available versions: {available_versions}"""
        )
    # filter models by version
    model_df = model_df[model_df["version"] == model_version]

    model_paths = []
    if model_dir is not None:
        model_dir = Path(model_dir)
    else:
        model_dir = get_model_data_dir()

    for _, row in model_df.iterrows():
        file_id = str(row["google_drive_id"])

        model_dir.mkdir(exist_ok=True)
        destination = model_dir / str(row["file_name"])
        timm_model_name = row["timm_model_name"]

        if not destination.exists() or force_download:
            download_file(file_id=file_id, destination=destination, source=source)

        elif destination.stat().st_size <= 1024 * 1024:
            download_file(file_id=file_id, destination=destination, source=source)

        model_paths.append({"Path": destination, "timm_model_name": timm_model_name})
    return model_paths
