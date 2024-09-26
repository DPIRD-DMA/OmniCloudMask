import gdown
import requests
import zipfile
from pathlib import Path
import logging


def download_file_from_google_drive(file_id: str, destination: Path) -> None:
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(destination), quiet=False)


def download_maxar_data(test_data_dir):
    maxar_url = "https://maxar-opendata.s3.us-west-2.amazonaws.com/events/Emilia-Romagna-Italy-flooding-may23/ard/32/120000303231/2023-05-23/1050010033C95B00-ms.tif"
    maxar_path = test_data_dir / "maxar.tif"
    if not maxar_path.exists():
        with requests.get(maxar_url, stream=True) as response:
            with open(maxar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)


def download_test_data():
    logging.info("Downloading test data...")
    test_data_dir = Path(__file__).parent / "test data"
    test_data_dir.mkdir(exist_ok=True)
    test_data_liks = {
        "LC81960302014022LGN00": "1ewmbD2YzxUS2IibMW5GTbcQyZIoz0TNf",
        "S2B_MSIL1C_20180302T150259_N0206_R125_T22WES_20180302T183800.SAFE": "1pGu_RdboqYcK4Q6_kjpnynCSzmNdUgcW",
        "S2A_MSIL2A_20170725T142751_N9999_R053_T19GBQ_20240410T040247.SAFE": "1ZEfXnNpWi75OV6fVhNvzbe6MhxsvXSI3",
    }

    for file_name, file_id in test_data_liks.items():
        zip_file = test_data_dir / f"{file_name}.zip"
        if not zip_file.exists():
            download_file_from_google_drive(file_id, zip_file)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(test_data_dir / file_name)

    download_maxar_data(test_data_dir)


if __name__ == "__main__":
    download_test_data()
    print("Test data setup complete.")
