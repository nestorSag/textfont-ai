from pathlib import Path
import logging
import typing as t
import inspect
import string
from argparse import Namespace
from functools import reduce

from pydantic import BaseModel, PositiveInt, PositiveFloat
import strictyaml as yml

from fontai.config.core import BaseConfigHandler, SimpleClassInstantiator, BasePipelineTransformConfig
from fontai.training.models import Model

import fontai.pipeline_stages as stages 

import tensorflow as tf

logger = logging.getLogger(__name__)

class Config(BasePipelineTransformConfig):
  """
  Wrapper class for the configuration of the MLPipeline class

  stage_sequence: list of ML stage classes along with their names and provided configuration

  """
  stage_sequence: t.List[MLPipelineStage]


class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for pipeline configuration processing logic.

  """

  def __init__(self):


    self.ALLOWED_STAGES = [getattr(stages, stage_class) for stage_class in stages.__all__ ]

    expected_schemas = [stage.get_config_handler().get_schema() for stage in self.STAGES_WITH_SCHEMA]

    expected_schema_union = reduce(lambda s1, s2: s1 | s2, expected_schemas)

    self.CONFIG_SCHEMA = {
    yml.Optional("writer_params", default = {}): self.IO_CONFIG_SCHEMA, 
    yml.Optional("reader_params", default = {}): self.IO_CONFIG_SCHEMA,
    "stages": yml.Seq(expected_schema_union)
    }


  def build_stage_sequence(self, stages_conf: yml.YAML):
    """
      Uses provided pipeline IO paths to impute missing IO paths for individual stages, and instatiates them
    """
    stage_input_conf = pipeline_input_conf

    for stage_conf in stages_conf:
      for stage_class in self.ALLOWED_STAGES:
        stage_schema = stage_class.get_config_handler().get_schema()
        try:
          stage_conf.revalidate(stage_schema)
          yield stage_class(stage_conf)
        except yml.exceptions.YAMLValidationError as e:
          pass


  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path = self.instantiate_io_handler(config.get("output_path"))
    input_path = self.instantiate_io_handler(config.get("input_path"))

    stage_sequence = list(self.build_stage_sequence(stages_yaml=config.get("stages")))   

    return Config(
      output_path = output_path, 
      input_path = input_path, 
      stage_sequence = stage_sequence,
      yaml = config)
