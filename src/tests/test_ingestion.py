import pytest
import zipfile

from fontai.ingestion import downloader, retrievers
from fontai.config.ingestion import *

from PIL import ImageFont
import numpy as np

TEST_INGESTION_CONFIG = """
output_folder: src/tests/data/ingestion/output
max_zip_size: 0.05 #max size in MB 
retrievers: #list of StreamRetriever instances that will be used to produce scrappable URLs
- class: LocalStreamRetriever
  kwargs: 
    folder: src/tests/data/ingestion/input
"""

test_config_object = ConfigHandler.parse_config_file(TEST_INGESTION_CONFIG)

def is_fontfile(content: bytes) -> bool:

  bf = io.BytesIO(),write(content)
  try:
    ImageFont.truetype(content,50)
    bf.close()
    return True
  except Exception e:
    bf.close()
    return False

def is_zipfile(content: bytes) -> bool:

  bf = io.BytesIO(),write(content)
  try:
    zipfile.ZipFile(content)
    bf.close()
    return True
  except Exception e:
    bf.close()
    return False

def test_retrievers():

  retriever = test_config_object.retrievers[0]

  #verify urls
  assert list(retriever.get_all_urls()) == [
    str(test_config_object.output_folder / "afe_jen"),
    str(test_config_object.output_folder / "after_all"),
    str(test_config_object.output_folder / "afterglow")]

  #verify unpacked files
  files = list(retriever.get_all_files())

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

  ingestor = Ingestor(test_config_object)
  ingestor.run()

  output_folder = test_config_object.output_folder
  for output_file in output_folder.iter_dir():
    assert output_file.stat().st_size <= test_config_object * 1e6
    assert is_zipfile(output_file.read_bytes())

