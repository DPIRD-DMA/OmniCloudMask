from importlib import resources
from pathlib import Path
from typing import Union
import os
import sys

import csv
from gcp_utils import DataFile

from .__version__ import __version__ as omnicloudmask_version

GCS_MODEL_BUCKET = "gs://overstory-satellite-imagery/models/omnicloudmask"


def download_file(file_name: str, destination: Path) -> None:
    """Download a model file from GCS to local destination.

    Args:
        file_name (str): The name of the file in the GCS bucket.
        destination (Path): The local path where the file should be saved.
    """
    gcs_path = f"{GCS_MODEL_BUCKET}/{file_name}"
    gcs_file = DataFile(gcs_path)
    local_file = DataFile(str(destination))
    gcs_file.copy_to(local_file, overwrite=True)


def get_model_data_dir() -> Path:
    """Get the user data directory for model files"""
    if sys.platform == "win32":
        base = Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    data_dir = base / "omnicloudmask" / omnicloudmask_version
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_models(
    force_download: bool = False,
    model_dir: Union[str, Path, None] = None,
    model_version: float = 3.0,
) -> list[dict]:
    """
    Downloads the model weights from GCS and saves them locally.

    Args:
        force_download (bool): Whether to force download the model weights even if they
        already exist locally.
        model_dir (Union[str, Path, None]): The directory where the model weights
        should be saved.
        model_version (float): The version of the model weights to download.
    """

    with (resources.files("omnicloudmask") / "model_download_links.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Get available versions
    available_versions = sorted(set(float(row["version"]) for row in rows))
    if model_version not in available_versions:
        raise ValueError(
            f"Model version {model_version} not found. "
            f"Available versions: {available_versions}"
        )
    # Filter models by version
    filtered_rows = [row for row in rows if float(row["version"]) == model_version]

    model_paths = []
    if model_dir is not None:
        model_dir = Path(model_dir)
    else:
        model_dir = get_model_data_dir()

    for row in filtered_rows:
        file_name = row["file_name"]

        model_dir.mkdir(exist_ok=True)
        destination = model_dir / file_name
        timm_model_name = row["timm_model_name"]

        if not destination.exists() or force_download:
            download_file(file_name=file_name, destination=destination)

        elif destination.stat().st_size <= 1024 * 1024:
            download_file(file_name=file_name, destination=destination)

        model_paths.append({"Path": destination, "timm_model_name": timm_model_name})
    return model_paths
