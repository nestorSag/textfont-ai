import sys
import pytest
from pathlib import Path
import os

import fontai.io.scrappers as scrapper_module
from fontai.io.formats import InMemoryZipHolder
from fontai.io.records import ScoredLabeledChar
from fontai.config.pipeline import Config as PipelineConfig
from fontai.pipeline.stages import FontIngestion, LabeledExampleExtractor, Predictor
from fontai.pipeline.pipeline import Pipeline

import logging

sys.path.append("src/tests/")

from config_builders import full_processing_config_str, full_prediction_config_str, ingestion_config_str

import tensorflow as tf

import strictyaml as yml

def test_ingestion():
  config = FontIngestion.parse_config_str(ingestion_config_str)
  FontIngestion.run_from_config_object(config)

  assert [obj.name for obj in list(Path(config.output_path).iterdir())] == [obj.name for obj in list(Path(config.scrappers[0].folder).iterdir())]


@pytest.mark.parametrize("config_str", [
  full_processing_config_str(output_record_class = "LabeledChar"), 
  full_processing_config_str(output_record_class = "LabeledFont")])
def test_preprocessing(config_str):

  # do stream processing
  config = LabeledExampleExtractor.parse_config_str(config_str)

  data_path = list(Path(config.input_path).iterdir())[0]
  extractor = LabeledExampleExtractor.from_config_object(config)
  data = InMemoryZipHolder(filename = "0", content = data_path.read_bytes())
  output = list(extractor.transform(data))
  n_examples = len(output)
  assert sum([example.__class__.__name__ ==  config.output_record_class.__name__ for example in output]) == n_examples

  # do batch processing
  if os.path.exists(config.output_path):
    for file in Path(config.output_path).iterdir():
      os.remove(str(file))

  LabeledExampleExtractor.run_from_config_object(config)
  output_files = list(Path(config.output_path).iterdir())
  assert len(output_files) == 1




@pytest.mark.parametrize(
  "config_str", 
  [
    full_prediction_config_str(
      input_record_class = "LabeledChar",
      model = "Sequential"),
    full_prediction_config_str(
      input_record_class = "LabeledChar",
      model = "CharStyleSAAE"),
    full_prediction_config_str(
      input_record_class = "LabeledFont",
      model = "Sequential"),
    full_prediction_config_str(
      input_record_class = "LabeledFont",
      model = "FontStyleSAAE")
  ])
def test_predictor(config_str):
  #preemtively clean output folder
  config = Predictor.parse_config_str(config_str)
  for file in list(Path(config.output_path).iterdir()):
    os.remove(str(file))
  Predictor.fit_from_config_object(config)
  Predictor.fit_from_config_object(config, load_from_model_path = True)
  Predictor.run_from_config_object(config)

  assert True


# @pytest.mark.parametrize("ingestion_config_str, processing_config_str, predictor_config_str",
#   [
#     (
#       ingestion_config_str,
#       full_processing_config_str(output_record_class = "LabeledChar"),
#       full_prediction_config_str(
#       input_record_class = "LabeledChar",
#       model = "Sequential")
#       )
#   ])
# def test_pipeline(ingestion_config_str, processing_config_str, predictor_config_str):

#   classes = [FontIngestion, LabeledExampleExtractor, Predictor]
#   config_strs = [ingestion_config_str, processing_config_str, predictor_config_str]

#   configs = [cls.parse_config_str(config_str) for cls, config_str in zip(classes, config_strs)]




#   dummy_yaml = yml.as_document({"a": 1})
#   config = PipelineConfig(stages=classes, configs=configs, yaml=dummy_yaml)

#   Pipeline.run_from_config_object(config)
#   Pipeline.fit_from_config_object(config)
#   pipeline = Pipeline(classes, configs)




#   streaming_input_file = list(Path(configs[0].output_path).iterdir())[0]

#   data = InMemoryZipHolder(filename = "0", content = streaming_input_file.read_bytes())

#   out = list(pipeline.transform(data))

#   assert len(out) == 124
#   for elem in out:
#     assert isinstance(elem, ScoredLabeledChar)

#   assert True




