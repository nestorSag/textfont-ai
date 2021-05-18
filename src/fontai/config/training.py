from pathlib import Path
import logging
import typing as t
import inspect
import string
from argparse import Namespace
from functools import reduce

from pydantic import BaseModel, PositiveInt, PositiveFloat
import strictyaml as yml

from fontai.core import DataPath
from fontai.config.core import BaseConfigHandler, SimpleClassInstantiator
from fontai.training.models import Model

import tensorflow as tf

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):

  batch_size: PositiveInt
  epochs: PositiveInt
  steps_per_epoch: PositiveInt
  optimizer: tf.keras.optimizers.Optimizer
  loss: tf.keras.losses.Loss
  #lr_shrink_factor: PositiveFloat

  @classmethod
  def from_yaml(cls, yaml):
    schema_handler = SimpleClassInstantiator()
    args = yaml.data
    args["optimizer"] = schema_handler.get_instance(yaml.get("optimizer"))
    args["loss"] = schema_handler.get_instance(yaml.get("loss"))
    return TrainingConfig(**args)


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


class ModelFactory(object):
  """
    Factory class for ML models that takes YAML configuration objects

    """

  def __init__(self):

    self.SEQUENTIAL_MODEL_SCHEMA = yml.Map({
      "class": "Sequential",
      "layers": yml.Seq(self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)
      })

    self.MULTI_SEQUENTIAL_MODEL_SCHEMA = yml.Map({
      "class": yml.Str(),
      "kwargs": yml.MapPattern(
        yml.Str(), 
        self.SEQUENTIAL_MODEL_SCHEMA | self.ANY_PRIMITIVE_SCHEMA,
        )
      })

    self.PATH_TO_SAVED_MODEL_SCHEMA = yml.Map({"path": yml.Str()})

    self.schema_constructors = {
      BaseConfigHandler().PY_CLASS_INSTANCE_FROM_YAML_SCHEMA: ("SIMPLE PY CLASS", self.from_simple_python_class)
      self.PATH_TO_SAVED_MODEL_SCHEMA: ("SAVED MODEL PATH", self.from_path),
      self.SEQUENTIAL_MODEL_SCHEMA: ("KERAS SEQUENTIAL", self.from_keras_sequential),
      self.MULTI_SEQUENTIAL_MODEL_SCHEMA: ("MULTI SEQUENTIAL", self.from_multi_sequential)
    }

    self.MODEL_CONFIG_SCHEMA = reduce(lambda schema1, schema2: schema1 | schema2, list(self.schema_constructors.keys()))

  def from_yaml(self, model_yaml: yml.YAML):
    """
    Instantiate a ML model from a YAML object that contains the model's specification

    model_yaml: YAML object

    Returns an instance of class Model

    """
    for schema in self.schema_constructors:
      name, constructor = self.schema_constructors[schema]
      try:
        model_yaml.revalidate(schema)
        logger.info(f"Model schema matched to: {name}")
        model = constructor(model_yaml)
        return model
      except Exception as e:
        logger.debug(f"Model schema did not match {name}; {e}")
    raise Exception("No valid schema matched provided model YAML; look at DEBUG log level for more info.")


  def from_simple_python_class(self, model_yaml):
    """
    Instantiate a ML model from a YAML object that matches the  schema for a simple Python class (i.e. only primitive types arguments)

    model_yaml: YAML object

    Returns an instance of class Model

    """
    return Model(BaseConfigHandler.)

  def from_path(self,model_yaml):
    """
    Loads a saved model 

    model_yaml: YAML object

    Returns an instance of class Model

    """
    return Model.from_path(model_yaml.get("path").text)

  def from_keras_sequential(self, model_yaml):
    """
    Instantiate a ML model of Keras' Sequential class

    model_yaml: YAML object

    Returns an instance of class Model

    """
    layer_instances = [self.get_instance(layer_yaml) for layer_yaml in model_yaml.get("layers")]
    return Model(Sequential(layer_instances))

  def from_multi_sequential(self, model_yaml):
    """
    Instantiate a ML model that uses multiple Keras Sequential models internally

    model_yaml: YAML object

    Returns an instance of class Model

    """
    args = model_yaml.get("kwargs")
    materialised_kwargs = copy.deepcopy(args.data)
    for named_param in materialised_kwargs:
      try:
        args.get(named_param).revalidate(self.SEQUENTIAL_MODEL_SCHEMA)
        materialised_kwargs[named_param] = self.from_keras_sequential(self,args.get(named_param))
      except Exception as e:
        logger.debug(f"Parameter {named_param} does not match Sequential model schema.")
    return Model(globals()[args.get("class").text](**materialised_kwargs))



class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for training configuration processing logic.

  """

  def other_setup(self):
    self.model_factory = ModelFactory()

  def get_config_schema(self):

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

    self.DATA_PREPROCESSING_SCHEMA = {"filters": yml.Seq(self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)} | yml.EmptyList()

    schema = yml.Map({
      "output_path": yml.Str(), 
      "input_path": yml.Str(), 
      "training": self.TRAINING_CONFIG_SCHEMA,
      "model": self.model_factory.MODEL_CONFIG_SCHEMA,
      yml.Optional("preprocessing_filters", default = []): self.DATA_PREPROCESSING_SCHEMA 
       })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path = DataPath(config.data["output_path"])
    input_path = DataPath(config.data["input_path"])
    model = self.model_factory.from_yaml(config.get("model"))
    training_config = TrainingConfig.from_yaml(config.get("training"))

    return Config(
      output_path = output_path, 
      input_path = input_path, 
      model = model,
      training_config = training_config,
      yaml = config)
