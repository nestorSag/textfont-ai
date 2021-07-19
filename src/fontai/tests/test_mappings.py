
import os
import sys
import pytest
from pathlib import Path
import logging
from functools import reduce

from fontai.runners.stages import Preprocessing
from fontai.io.storage import BytestreamPath
from fontai.io.formats import InMemoryZipHolder, InMemoryFontfileHolder
from fontai.io.records import LabeledChar, LabeledFont

from fontai.preprocessing.mappings import InputToFontFiles, FontFileToLabeledChars, FeatureCropper, FeatureResizer, PipelineFactory

import numpy as np

sys.path.append("src/fontai/tests/")

from config_builders import full_processing_config_str

@pytest.mark.parametrize("input_file, processing_config, output_schemas", [
    (
      "src/tests/data/raw/afe_jen",
      Preprocessing.parse_config_str(full_processing_config_str("LabeledChar")),
      (LabeledChar, LabeledFont))
  ])


def test_mappings(input_file, processing_config, output_schemas):
  # preemtively clean output folder
  # for file in list(Path(processing_config.output_path).iterdir()):
  #   os.remove(str(file))


  # parse basic character-wise transformations
  data = InMemoryZipHolder(filename = "0", content = Path(input_file).read_bytes())

  font_files = list(InputToFontFiles().map(data))
  assert [file.filename for file in font_files] == ["AFE_Jen.ttf", "AFE_Jen_Bold.ttf"]
  assert sum([file.__class__.__name__ == "InMemoryFontfileHolder" for file in font_files]) == 2

  extractor = FontFileToLabeledChars(**processing_config.font_to_array_config.dict())
  examples = reduce(lambda a, b: list(extractor.map(a)) + list(extractor.map(b)), font_files)
  n_examples = len(examples)

  extraction_shape = processing_config.font_to_array_config.canvas_size
  assert sum([example.__class__.__name__ == "LabeledChar" for example in examples]) == n_examples
  assert sum([example.features.shape == (extraction_shape, extraction_shape) for example in examples]) == n_examples
  assert sum([example.features.dtype == np.uint8 for example in examples]) == n_examples
  assert sum([np.any(example.features > 0) for example in examples]) == n_examples

  cropper = FeatureCropper()
  cropped = [list(cropper.map(example))[0] for example in examples]
  assert sum([np.any(x.features[0,:]) > 0 and np.any(x.features[:,0] > 0) for x in cropped]) == n_examples

  out_size = processing_config.output_array_size
  resizer = FeatureResizer(out_size)
  resized = [list(resizer.map(example))[0] for example in cropped]
  assert sum([x.features.shape ==  (out_size, out_size) for x in resized])

  ## test pipelines
  for output_schema in output_schemas:
    pipeline = PipelineFactory.create(
    output_record_class = output_schema,
    output_array_size = processing_config.output_array_size,
    **processing_config.font_to_array_config.dict())

    for output in pipeline.map(data):
      assert isinstance(output, output_schema)