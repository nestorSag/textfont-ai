import pytest
import zipfile
import io
from pathlib import Path

from fontai.ingestion import bundler, scrappers
from fontai.config.ingestion import ConfigHandler

from PIL import ImageFont
import numpy as np

TEST_INGESTION_CONFIG = """
output_path: src/tests/data/ingestion/output
max_zip_size: 0.2 #max size in MB 
input_path: src/tests/data/ingestion/input
"""

test_config_object = ConfigHandler().from_string(TEST_INGESTION_CONFIG)

def is_fontfile(content: bytes) -> bool:

  #bf = io.BytesIO()
  #bf.write(content)
  try:
    ImageFont.truetype(io.BytesIO(content),50)
    return True
  except Exception as e:
    return False

def is_zipfile(content: bytes) -> bool:

  try:
    zipfile.ZipFile(io.BytesIO(content))
    return True
  except Exception as e:
    return False

def test_local_file_scrapper():

  scrapper = test_config_object.input_path

  #verify urls
  assert scrapper.list_files() == [
    scrapper.path / "afe_jen",
    scrapper.path / "after_fall",
    scrapper.path / "afterglow"]

  #verify unpacked files
  files = list(scrapper.get_files())

  expected_filenames = [
    "AFE_Jen.ttf",
    "AFE_Jen_Bold.ttf",
    "After Fall.otf",
    "afterglow.ttf"]

  actual_filenames = [file.filename for file in files]

  assert actual_filenames == expected_filenames

  actual_file_contents = [file.content for file in files]

  assert np.all([is_fontfile(file) for file in actual_file_contents])


def test_ingestor():

  ingestor = downloader.Ingestor(test_config_object)
  ingestor.run()

  output_path = test_config_object.output_path
  for output_file in output_path.list_files():
    assert Path(str(output_file)).stat().st_size <= test_config_object.max_zip_size * 1e6
    assert is_zipfile(output_file.read_bytes())

