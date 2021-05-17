import pytest
from pathlib import Path

import numpy as np
from fontai.core import DataPath, InMemoryFile, LabeledExample, KeyValuePair, TfrHandler
from fontai.preprocessing.file_processing import *
from fontai.config.preprocessing import ConfigHandler

from tensorflow.data import TFRecordDataset
from  tensorflow.python.data.ops.dataset_ops import MapDataset

INPUT_PATH = DataPath("src/tests/data/ingestion/output/0")
OUTPUT_PATH = DataPath("src/tests/data/preprocessing/output")

TEST_PROCESSING_CONFIG = """
output_path: src/tests/data/preprocessing/output
input_path: src/tests/data/ingestion/output
output_array_size: 64
font_extraction_config:
  font_extraction_size: 100
  canvas_size: 500
  canvas_padding: 100
beam_cmd_line_args:
-  --runner
-  DirectRunner
"""

Path(str(OUTPUT_PATH)).mkdir(parents=True, exist_ok=True)

test_config_object = ConfigHandler().from_string(TEST_PROCESSING_CONFIG)


def test_stages():

  transformer = DataPathReader()
  test_object = list(transformer.map(INPUT_PATH))[0]
  zip_bytes = Path(str(INPUT_PATH)).read_bytes()
  assert test_object == KeyValuePair(key="0",value=InMemoryFile(filename="0",content=zip_bytes))


  transformer = ZipToFontFiles()
  test_object = list(transformer.map(test_object))[0]
  zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))
  file_name = zip_file.namelist()[0]
  file_bytes = zip_file.read(file_name)
  assert  test_object == KeyValuePair(key="0", value = InMemoryFile(filename=file_name,content=file_bytes))


  transformer = FontFileToCharArrays(charset = "a", canvas_size = 100, canvas_padding = 20, font_size = 20)
  test_object = list(transformer.map(test_object))[0]
  overall_nonzero = np.sum(test_object.value.x > 0)
  assert test_object.key == "0"
  assert test_object.value.y == "a"
  assert isinstance(test_object.value.x, np.ndarray)
  assert test_object.value.x.shape == (100,100)
  assert np.any(test_object.value.x > 0)
  assert test_object.value.x.dtype == np.uint8


  transformer = ArrayCropper()
  test_object = list(transformer.map(test_object))[0]
  assert test_object.key == "0"
  assert test_object.value.y == "a"
  assert np.sum(test_object.value.x > 0) == overall_nonzero
  assert np.all(np.zeros((2,)) < test_object.value.x.shape)
  assert np.all(100*np.ones((2,)) >= test_object.value.x.shape)


  transformer = ArrayResizer(output_size=64)
  test_object = list(transformer.map(test_object))[0]
  assert test_object.key == "0"
  assert test_object.value.y == "a"
  assert test_object.value.x.shape == (64,64)


  transformer = TfrRecordWriter(OUTPUT_PATH)
  transformer.process(test_object)
  transformer.close()
  records = TFRecordDataset(filenames=str(OUTPUT_PATH/ "0.tfr")).map(TfrHandler().from_tfr)
  record = iter(records).next()
  assert record["label"] == bytes("a".encode("utf-8"))

def test_beam_pipeline():

  processor = FileProcessor(test_config_object)
  processor.run()

  output_path = test_config_object.output_path
  for output_file in output_path.list_files():
    record = TFRecordDataset(filenames=str(output_file)).map(TfrHandler().from_tfr)
    assert isinstance(record, MapDataset)





