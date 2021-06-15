from pathlib import Path
import logging
import typing as t
import inspect
import traceback
import string
from argparse import Namespace
import copy
from functools import reduce

from pydantic import BaseModel, PositiveInt, PositiveFloat, validator
import strictyaml as yml

from fontai.config.core import BaseConfigHandler, SimpleClassInstantiator, BasePipelineTransformConfig
import fontai.prediction.input_processing as input_processing
import fontai.prediction.models as custom_models

from tensorflow import keras
from tensorflow.keras import layers


logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):

  batch_size: PositiveInt
  epochs: PositiveInt
  steps_per_epoch: PositiveInt
  optimizer: keras.optimizers.Optimizer
  loss: keras.losses.Loss
  charset: str = "all"
  filters: t.List[t.Callable] = []
  seed: int = 1

  @validator("charset")
  def allowed_charsets(charset):
    allowed_vals = ["all","uppercase","lowercase","digits"]
    if charset in allowed_vals:
      return charset
    else:
      raise ValueError(f"charset must be one of {allowed_vals}")
  #lr_shrink_factor: PositiveFloat

  @classmethod
  def from_yaml(cls, yaml):
    schema_handler = SimpleClassInstantiator()
    args = yaml.data
    args["optimizer"] = schema_handler.get_instance(yaml=yaml.get("optimizer"), scope=keras.optimizers)
    args["loss"] = schema_handler.get_instance(yaml=yaml.get("loss"), scope=keras.losses)
    if  yaml.get("filters") is not None:
      args["filters"] = [getattr(input_processing, subyaml.get("name").text)(**subyaml.get("kwargs").data) for subyaml in yaml.get("filters").data]
    else:
      args["filters"] = []
    
    return TrainingConfig(**args)

  class Config:
    arbitrary_types_allowed = True


class Config(BasePipelineTransformConfig):
  """
  Wrapper class for the configuration of the ModelTrainingStage class

  training_config: Runtime configuration for training routine

  model: Model instance that's going to be trained

  """
  training_config: TrainingConfig
  model_path: str
  model: keras.Model

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True


class ModelFactory(object):
  """
    Factory class for ML models that takes YAML configuration objects

    """

  def __init__(self):

    self.yaml_to_obj = SimpleClassInstantiator()

    self.SEQUENTIAL_MODEL_SCHEMA = yml.Map({
      "class": yml.Str(),
      "kwargs": yml.Map({"layers": yml.Seq(self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)})
      })

    self.MULTI_SEQUENTIAL_MODEL_SCHEMA = yml.Map({
      "class": yml.Str(),
      "kwargs": yml.MapPattern(
        yml.Str(), 
        self.yaml_to_obj.ANY_PRIMITIVES | self.SEQUENTIAL_MODEL_SCHEMA,
        )
      })

    self.PATH_TO_SAVED_MODEL_SCHEMA = yml.Map({"path": yml.Str(), yml.Optional("custom_class", default = None): yml.Str() | yml.EmptyNone()})

    self.schema_constructors = {
      self.PATH_TO_SAVED_MODEL_SCHEMA: ("SAVED MODEL PATH", self.from_path),
      self.SEQUENTIAL_MODEL_SCHEMA: ("KERAS SEQUENTIAL", self.from_keras_sequential),
      self.MULTI_SEQUENTIAL_MODEL_SCHEMA: ("MULTI SEQUENTIAL", self.from_multi_sequential)
    }

    #self.MODEL_CONFIG_SCHEMA = reduce(lambda schema1, schema2: schema1 | schema2, list(self.schema_constructors.keys()))

  def from_yaml(self, yaml: yml.YAML):
    """
    Instantiate a ML model from a YAML object that contains the model's specification

    model_yaml: YAML object

    Returns an instance of class Model

    """
    for schema in self.schema_constructors:
      name, constructor = self.schema_constructors[schema]
      try:
        model_yaml = yml.load(yaml.as_yaml(), schema)
        logger.info(f"Model schema matched to: {name}")
        model = constructor(model_yaml)
        return model
      except Exception as e:
        logger.debug(f"Model schema did not match {name}; {e}")
    raise Exception("No valid schema matched provided model YAML; look at DEBUG log level for more info.")

  def from_path(self,model_yaml):
    """
    Loads a saved model 

    model_yaml: YAML object

    Returns an instance of class Model

    """
    model_class = model_yaml.get("custom_class")
    custom_objects = {model_class.text: getattr(custom_models, model_class.text)} if model_class is not None else None

    return keras.models.load_model(model_yaml.get("path").text, custom_objects = custom_objects)

  def from_keras_sequential(self, model_yaml):
    """
    Instantiate a ML model of Keras' Sequential class

    model_yaml: YAML object

    Returns an instance of class Model

    """
    model_layers = model_yaml.get("kwargs").get("layers")
    layer_instances = [self.yaml_to_obj.get_instance(layer_yaml, layers) for layer_yaml in model_layers]
    return keras.Sequential(layer_instances)

  def from_multi_sequential(self, model_yaml):
    """
    Instantiate a ML model that uses multiple Keras Sequential models internally

    model_yaml: YAML object

    Returns an instance of class Model

    """
    args = model_yaml.get("kwargs")
    materialised_kwargs = copy.deepcopy(args.data)
    for arg in args:
      try:
        yml.load(args.get(arg).as_yaml(), self.SEQUENTIAL_MODEL_SCHEMA)
        materialised_kwargs[arg] = self.from_keras_sequential(args.get(arg))
      except Exception as e:
        logger.debug(f"Parameter {arg} does not match Sequential model schema.")
    return getattr(custom_models, model_yaml.get("class").text)(**materialised_kwargs)


class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for training configuration processing logic.

  """

  def other_setup(self):
    self.model_factory = ModelFactory()

  def get_config_schema(self):

    self.DATA_PREPROCESSING_SCHEMA = yml.Seq(self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA) | yml.EmptyNone()

    self.TRAINING_CONFIG_SCHEMA = yml.Map({
      "batch_size": yml.Int(),
      "epochs": yml.Int(),
      "metrics": yml.Seq(yml.Str()),
      "loss": self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA,
      yml.Optional(
        "steps_per_epoch", 
        default = 10000): yml.Int(),
      yml.Optional(
        "optimizer", 
        default = {"class": "Adam", "kwargs": {}}): self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA,
      yml.Optional(
        "filters",
        default = None): self.DATA_PREPROCESSING_SCHEMA
    })

    schema = yml.Map({
      yml.Optional("input_path", default = None): self.IO_CONFIG_SCHEMA, 
      yml.Optional("output_path", default = None): self.IO_CONFIG_SCHEMA,
      "model_path": self.IO_CONFIG_SCHEMA,
      "training": self.TRAINING_CONFIG_SCHEMA,
      "model": yml.Any(),
      yml.Optional("charset", default = "uppercase"): yml.Str()
       })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    input_path, output_path = config.get("input_path").text, config.get("output_path").text

    model_path = config.get("model_path").text
    model = self.model_factory.from_yaml(config.get("model"))
    training_config = TrainingConfig.from_yaml(config.get("training"))

    return Config(
      input_path = input_path, 
      output_path = output_path,
      model_path = model_path,
      model = model,
      training_config = training_config,
      yaml = config)
