from pathlib import Path
import logging
import typing as t
import inspect
import string
from argparse import Namespace

from pydantic import BaseModel, PositiveInt, PositiveFloat
import strictyaml as yml

from fontai.core import DataPath
from fontai.config.core import BaseConfigHandler

import tensorflow as tf
logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):

  batch_size: PositiveInt
  epochs: PositiveInt
  steps_per_epoch: PositiveInt
  optimizer: tf.keras.optimizers.Optimizer
  loss: tf.keras.losses.Loss
  lr_shrink_factor: PositiveFloat


class Config(BaseModel):
  """
  Wrapper class for the configuration of the ImageExtractor class

  output_path: folder in which scrapped and zipped ttf/otf files will be saved

  max_zip_size: maximum pre-compression size of zipped output files

  scrappers: list of FileScrapper instances from which scrapped files will be processed.

  """
  input_path: DataPath
  output_path: DataPath
  training_config: TrainingConfig
  model: Model
  yaml: yml.YAML

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

    
class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for training configuration processing logic.

  """

  def __init__(self):
    
    self.SEQUENTIAL_MODEL_SCHEMA = yml.Map({
      "class": "tf.keras.Sequential",
      "layers": yml.Seq(self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)
      })

    self.MULTI_SEQUENTIAL_MODEL_SCHEMA = yml.Map({
      "class": yml.Str(),
      "kwargs": yml.MapPattern(
        yml.Str(), 
        self.SEQUENTIAL_MODEL_SCHEMA | self.ANY_PRIMITIVE_SCHEMA,
        )
      })

    self.MODEL_CONFIG_SCHEMA = 
      self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA | 
      yml.Map({"path": yml.Str()}) | 
      self.SEQUENTIAL_MODEL_SCHEMA | 
      self.MULTI_SEQUENTIAL_MODEL_SCHEMA

    self.TRAINING_CONFIG_SCHEMA = yml.Map({
      "batch_size": yml.Int(),
      "epochs": yml.Int(),
      Optional(
        "steps_per_epoch", 
        default = 10000): yml.Int(),
      Optional(
        "optimizer", 
        default = {"class": "Adam", "kwargs": {}}): self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA
    })

    self.DATA_PREPROCESSING_SCHEMA = {"filters": yml.Seq(self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)}

    self.CONFIG_SCHEMA = yml.Map({
      "output_path": yml.Str(), 
      "input_path": yml.Str(), 
      "training": self.TRAINING_CONFIG_SCHEMA,
      "model": self.MODEL_CONFIG_SCHEMA,
      "preprocessing_filters": self.DATA_PREPROCESSING_SCHEMA
       })

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path = DataPath(config.data["output_path"])
    input_path = DataPath(config.data["input_path"])

    return Config(
      output_path = output_path, 
      input_path = input_path, 
      output_array_size = output_array_size,
      font_to_array_config = f2a_config,
      beam_cmd_line_args = beam_cmd_line_args,
      yaml = config)
