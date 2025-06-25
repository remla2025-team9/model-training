"""
Downloads raw data from an external source (e.g., Google Drive)
and saves it to the raw data directory.
"""
import logging
import gdown
from . import config

GOOGLE_DRIVE_FILE_ID = "1HeVAtmRY2lCydCabmsi2nZoW1BVou3p9"

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_raw_data(file_id, output_path):
    """
    Downloads a file from Google Drive given its file ID.

    Args:
        file_id (str): The public file ID from the Google Drive share link.
        output_path (pathlib.Path): The local path to save the downloaded file.
    """
    logger.info(f"Downloading raw data from Google Drive to: {output_path}")

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct the URL that gdown uses
    url = f'https://drive.google.com/uc?id={file_id}'

    try:
        gdown.download(url, str(output_path), quiet=False)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download file. Error: {e}")
        raise RuntimeError(f"Failed to download file with ID {file_id}.") from e


def main():
    """
    Main function to run the data downloading process.
    """
    logger.info("--- Starting raw data download process ---")

    # The output path is defined in your project's config file
    output_file_path = config.RAW_DATA_DIR / "training_data.tsv"

    download_raw_data(file_id=GOOGLE_DRIVE_FILE_ID, output_path=output_file_path)

    logger.info("--- Raw data download process finished ---")


if __name__ == "__main__":
    main()
