from pathlib import Path

import pytest
import strictyaml as yml
from fontai.pipeline.stages import Predictor, FontIngestion, LabeledExampleExtractor
from fontai.pipeline.pipeline import Pipeline
from fontai.config.pipeline import Config
from fontai.io.formats import InMemoryZipHolder
from fontai.io.records import ScoredLabeledExample

STREAMING_DATA_TEST_PATH = "src/tests/data/ingestion/input/afe_jen"

INGESTION_CONFIG = """
scrappers:
- class: LocalScrapper
  kwargs: 
    folder: src/tests/data/ingestion/input
output_path: src/tests/data/ingestion/output
"""

PREPROCESSING_CONFIG = """
input_path: src/tests/data/ingestion/output
output_path: src/tests/data/preprocessing/output
output_array_size: 64
max_output_file_size: 64
font_extraction_size: 100
canvas_size: 500
canvas_padding: 100
"""

PREDICTOR_CONFIG = """
input_path: src/tests/data/preprocessing/output
output_path: src/tests/data/prediction/output
model_path: src/tests/data/prediction/model
training:
  batch_size: 32
  epochs: 10
  steps_per_epoch: 10
  optimizer:
    class: Adam
  loss:
    class: CategoricalCrossentropy
  metrics:
  - accuracy
model:
  class: Sequential
  kwargs:
    layers:
    - class: Input
      kwargs:
        shape:
        - 64
        - 64
        - 1
    - class: Flatten
    - class: Dense
      kwargs: 
        units: 10
        activation: elu
    - class: Dense
      kwargs: 
        units: 62
        activation: sigmoid
"""


def test_predictor():
  classes = [FontIngestion, LabeledExampleExtractor, Predictor]
  config_strs = [INGESTION_CONFIG, PREPROCESSING_CONFIG, PREDICTOR_CONFIG]

  configs = [cls.parse_config_str(config_str) for cls, config_str in zip(classes, config_strs)]

  dummy_yaml = yml.as_document({"a": 1})
  config = Config(stages=classes, configs=configs, yaml=dummy_yaml)

  Pipeline.run_from_config_object(config)
  Pipeline.fit_from_config_object(config)
  pipeline = Pipeline(classes, configs)

  data = InMemoryZipHolder(filename = "0", content = Path(STREAMING_DATA_TEST_PATH).read_bytes())

  out = list(pipeline.transform(data))

  assert len(out) == 124
  for elem in out:
    assert isinstance(elem, ScoredLabeledExample)

  assert True








