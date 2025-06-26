import pytest
import tempfile
from unittest import mock
from src import dataset

def test_download_raw_data_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_output_path = dataset.config.Path(tmpdir) / "test.tsv"

        with mock.patch("gdown.download") as mock_download:
            mock_download.return_value = str(mock_output_path)
            dataset.download_raw_data(file_id="dummy_id", output_path=mock_output_path)

           
            mock_download.assert_called_once()
            url_called = mock_download.call_args[0][0]
            assert "https://drive.google.com/uc?id=dummy_id" in url_called

def test_download_raw_data_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_output_path = dataset.config.Path(tmpdir) / "test.tsv"

        with mock.patch("gdown.download", side_effect=Exception("Download failed")):
            with pytest.raises(RuntimeError, match="Failed to download file"):
                dataset.download_raw_data(file_id="dummy_id", output_path=mock_output_path)
