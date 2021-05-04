import pytest
import zipfile
import io
from pathlib import Path

from fontai.ingestion import downloader, scrappers
from fontai.config.ingestion import *

from PIL import ImageFont
import numpy as np

TEST_INGESTION_CONFIG = """
output_folder: src/tests/data/ingestion/output
max_zip_size: 0.2 #max size in MB 
scrappers: #list of FileScrapper instances that will be used to produce scrappable URLs
- class: LocalFileScrapper
  kwargs: 
    folder: src/tests/data/ingestion/input
"""

test_config_object = ConfigHandler.from_string(TEST_INGESTION_CONFIG)

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

def test_scrappers():

  scrapper = test_config_object.scrappers[0]

  #verify urls
  assert list(scrapper.get_sources()) == [
    scrapper.folder / "afe_jen",
    scrapper.folder / "after_fall",
    scrapper.folder / "afterglow"]

  #verify unpacked files
  files = list(scrapper.get_all_files())

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

  output_folder = test_config_object.output_folder
  for output_file in output_folder.iterdir():
    assert output_file.stat().st_size <= test_config_object.max_zip_size * 1e6
    assert is_zipfile(output_file.read_bytes())

