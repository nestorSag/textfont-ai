import os
from pathlib import Path
import logging
from functools import reduce

from fontai.pipeline.stages import LabeledExampleExtractor
from fontai.io.storage import BytestreamPath
from fontai.io.formats import InMemoryZipHolder, InMemoryFontfileHolder
from fontai.preprocessing.mappings import InputToFontFiles, FontFileToLabeledExamples, FeatureCropper, FeatureResizer

import numpy as np

TEST_PREPROCESSING_CONFIG = """
input_path: src/tests/data/ingestion/output
output_path: src/tests/data/preprocessing/output
output_array_size: 64
max_output_file_size: 64
font_extraction_size: 100
canvas_size: 500
canvas_padding: 100
"""

MAPPING_TEST_INPUT = BytestreamPath("src/tests/data/ingestion/output/afe_jen")

TEST_CONFIG_OBJECT = LabeledExampleExtractor.parse_config_str(TEST_PREPROCESSING_CONFIG)

logger = logging.getLogger(__name__)

# preemtively clean output folder
for file in list(Path(TEST_CONFIG_OBJECT.output_path).iterdir()):
  os.remove(str(file))


def test_mappings():
  data = InMemoryZipHolder(filename = "0", content = MAPPING_TEST_INPUT.read_bytes())

  font_files = list(InputToFontFiles().map(data))
  assert [file.filename for file in font_files] == ["AFE_Jen.ttf", "AFE_Jen_Bold.ttf"]
  assert sum([file.__class__.__name__ == "InMemoryFontfileHolder" for file in font_files]) == 2

  extractor = FontFileToLabeledExamples(**TEST_CONFIG_OBJECT.font_to_array_config.dict())
  examples = reduce(lambda a, b: list(extractor.map(a)) + list(extractor.map(b)), font_files)
  n_examples = len(examples)

  extraction_shape = TEST_CONFIG_OBJECT.font_to_array_config.canvas_size
  assert sum([example.__class__.__name__ == "LabeledExample" for example in examples]) == n_examples
  assert sum([example.features.shape == (extraction_shape, extraction_shape) for example in examples]) == n_examples
  assert sum([example.features.dtype == np.uint8 for example in examples]) == n_examples
  assert sum([np.any(example.features > 0) for example in examples]) == n_examples

  cropper = FeatureCropper()
  cropped = [list(cropper.map(example))[0] for example in examples]
  assert sum([np.any(x.features[0,:]) > 0 and np.any(x.features[:,0] > 0) for x in cropped]) == n_examples

  out_size = TEST_CONFIG_OBJECT.output_array_size
  resizer = FeatureResizer(out_size)
  resized = [list(resizer.map(example))[0] for example in cropped]
  assert sum([x.features.shape ==  (out_size, out_size) for x in resized])


def test_stream_preprocessing():

  extractor = LabeledExampleExtractor.from_config_object(TEST_CONFIG_OBJECT)
  data = InMemoryZipHolder(filename = "0", content = MAPPING_TEST_INPUT.read_bytes())
  output = list(extractor.transform(data))
  n_examples = len(output)
  assert sum([example.__class__.__name__ == "LabeledExample" for example in output]) == n_examples


def test_batch_preprocessing():
  
  LabeledExampleExtractor.run_from_config_object(TEST_CONFIG_OBJECT)
  output_files = list(Path(TEST_CONFIG_OBJECT.output_path).iterdir())
  assert len(output_files) == 1






