from pathlib import Path

import pandas as pd
import gdown


def download_file_from_google_drive(file_id: str, destination: Path) -> None:
    """
    Downloads a file from Google Drive and saves it at the given destination using gdown.

    Args:
        file_id (str): The ID of the file on Google Drive.
        destination (Path): The local path where the file should be saved.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(destination), quiet=False)


def get_models() -> list[Path]:
    """
    Downloads the model weights from Google Drive and saves them locally.
    """
    df = pd.read_csv(
        Path(__file__).resolve().parent / "models/model_download_links.csv"
    )
    model_paths = []

    for _, row in df.iterrows():
        file_id = str(row["google_drive_id"])
        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(exist_ok=True)
        destination = model_dir / str(row["file_name"])

        if not destination.exists():
            # print(f"Downloading {row['file_name']} to {destination}...")
            download_file_from_google_drive(file_id, destination)

        elif destination.stat().st_size <= 1024 * 1024:
            # print(f"Downloading {row['file_name']} to {destination}...")
            download_file_from_google_drive(file_id, destination)
        model_paths.append(destination)
    return model_paths
