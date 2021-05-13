import pytest
from libpath import Path

import numpy as np
from fontai.core import DataPath, InMemoryFile, LabeledExample, KeyValuePair, TfrHandler
from fontai.preprocessing.file_processing import *


INPUT_PATH = DataPath("src/tests/data/ingestion/output/0")
OUTPUT_PATH = DataPath("src/tests/data/preprocessing/output")

def test_stages():

  transformer = DataPathReader()
  test_object = list(transformer.map(INPUT_PATH))[0]
  zip_bytes = Path(str(INPUT_PATH)).read_bytes()
  assert test_object == KeyValuePair(key="0",value=zip_bytes)


  transformer = ZipToFontFiles()
  test_object = list(transformer.map(test_object))[0]
  zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))
  file_name = zip_file.namelist()[0]
  file_bytes = zip_file.read(file_name)
  assert  test_object == KeyValuePair(key="0", value = InMemoryFile(filename=file_name,content=file_bytes))


  transformer = FontFileToCharArrays(charset = "a", canvas_size = 100, canvas_padding = 20, font_Size = 20)
  test_object = list(transformer.map(test_object))[0]
  overall_nonzero = np.sum(test_object.value.x > 0)
  assert test_object.key == "0"
  assert test_object.value.y == "a"
  assert isinstance(test_object.value.x, np.ndarray)
  assert test_object.value.x.shape == (100,100)
  assert np.any(test_object.value.x > 0)
  assert test_object.value.x.dtype = np.uint8


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
  transformer.map(test_object)
  records = tf.data.TFRecordDataset(filenames=str(OUTPUT_PATH))
    .map(TfrHandler.from_tfr)
  record = iter(records).next()
  assert record["label"] == bytes("a".encode("utf-8"))

  #assert test_object.key == "0"
  #assert test_object.value.y == "a"
  #assert test_object.value.x.shape == (64,64)





