from pathlib import Path
import logging
import typing as t
import inspect
import string
from argparse import Namespace
from functools import reduce

from pydantic import BaseModel, PositiveInt, PositiveFloat
import strictyaml as yml

from fontai.core.base import BaseConfigHandler, SimpleClassInstantiator, BaseConfigs
from fontai.training.models import Model

from fontai.config.ingestion import Config as IngestionConfig, ConfigHandler as IngestionConfigHandler
from fontai.config.preprocessing import Config as PreprocConfig, ConfigHandler as PreprocConfigHandler
from fontai.config.training import Config as TrainingConfig, ConfigHandler as TrainingConfigHandler

import tensorflow as tf

logger = logging.getLogger(__name__)

class Config(BaseConfig):
  """
  Wrapper class for the configuration of the MLPipeline class

  stage_sequence: list of ML stage classes along with their names and provided configuration

  """
  stage_sequence: t.List[t.Tuple[str, MLPipelineStage, BaseConfig]]


class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for training configuration processing logic.

  """

  def __init__(self):

    def remove_io_from_handler_schema(handler):
      schema = handler.get_config_schema()
      key_subset = [key for key in schema._required_keys() if key not in ["input_path","output_path"]]
      schema._required_keys = key_subset
      return schema

    self.ALLOWED_STAGE_HANDLERS = {name: (handler, remove_io_from_handler_schema(handler)) for name, handler in 
      [
        ("ingestion", IngestionConfigHandler()),
        ("preprocessing", PreprocConfigHandler()),
        ("training", TrainingConfigHandler())
      ]
    }

    self.CONFIG_SCHEMA = {
    "input_path": self.IO_CONFIG_SCHEMA,
    "output_path": self.IO_CONFIG_SCHEMA,
    "stages": yml.Any()

    }

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path = self.instantiate_io_handler(config.get("output_path"))
    input_path = self.instantiate_io_handler(config.get("input_path"))
    model = self.model_factory.from_yaml(config.get("model"))
    training_config = TrainingConfig.from_yaml(config.get("training"))

    return Config(
      output_path = output_path, 
      input_path = input_path, 
      model = model,
      training_config = training_config,
      yaml = config)
