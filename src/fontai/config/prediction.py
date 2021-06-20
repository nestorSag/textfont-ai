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
import tensorflow.keras.callbacks as tf_callbacks
from tensorflow.random import set_seed

import fontai.prediction.callbacks as custom_callbacks

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):

  """
  Training configuration wrapper for a Predictor ML stage
  
  Args:
      batch_size (int): batch size
      epochs (int): epochs
      steps_per_epoch (int): batches per epoch
      optimizer (keras.optimizers.Optimizer): optimizer
      loss (keras.losses.Loss): loss function
      filters t.List[t.Callable]: list of model input filter functions from `input_processing` module
      seed (int) Tensorflow global random seed
      metrics (t.List[str], optional): list of metrics to display
      callbacks (t.List[tf_callbacks.Callback], optional): list of callbakcs to use at training time.
  """

  batch_size: PositiveInt
  epochs: PositiveInt
  steps_per_epoch: PositiveInt
  optimizer: keras.optimizers.Optimizer
  loss: keras.losses.Loss
  filters: t.List[t.Callable] = []
  seed: PositiveInt = 1
  metrics: t.Optional[t.List[str]] = None
  callbacks: t.Optional[t.List[tf_callbacks.Callback]] = None

  #lr_shrink_factor: PositiveFloat

  @classmethod
  def from_yaml(cls, yaml):
    """Instantiate from a yml.YAML object
    
    Args:
        yaml (yml.YAML): Input YAML object following the schema given by ConfigHandler.training_config_schema
    
    Returns:
        TrainingConfig: Initialised object
    """
    schema_handler = SimpleClassInstantiator()
    callback_factory = CallbackFactory()
    args = yaml.data

    # the following objects are not primitive types and need to be instantiated from YAML definitions
    args["optimizer"] = schema_handler.get_instance(yaml=yaml.get("optimizer"), scope=keras.optimizers)
    args["loss"] = schema_handler.get_instance(yaml=yaml.get("loss"), scope=keras.losses)
    if  yaml.get("filters") is not None:
      args["filters"] = [getattr(input_processing, subyaml.get("name").text)(**subyaml.get("kwargs").data) for subyaml in yaml.get("filters").data]
    else:
      args["filters"] = []
    
    if  yaml.get("callbacks") is not None:
      args["callbacks"] = [CallbackFactory.create(yaml) for yaml in yaml.get("callbacks")]
    else:
      args["callbacks"] = None

    return TrainingConfig(**args)

  class Config:
    arbitrary_types_allowed = True


class Config(BasePipelineTransformConfig):
  """
  Wrapper class for the configuration of the ModelTrainingStage class
  
  Args:
      training_config (TrainingConfig): Runtime configuration for training routine
      model (keras.Model): Model instance that's going to be trained
  
  """
  training_config: TrainingConfig
  model_path: str
  model: keras.Model
  charset: str

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

  @validator("charset")
  def allowed_charsets(charset: str):
    """Vlidator for charset attribute
    
    Args:
        charset (str): charset attribute
    
    Returns:
        str: validated charset
    
    Raises:
        ValueError: If charset is invalid
    """
    allowed_vals = ["all","uppercase","lowercase","digits"]
    if charset in allowed_vals:
      return charset
    else:
      raise ValueError(f"charset must be one of {allowed_vals}")


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

    self.PATH_TO_SAVED_MODEL_SCHEMA = yml.Map({"path": yml.Str(), yml.Optional("custom_class"): yml.Str()})

    self.schema_constructors = {
      self.PATH_TO_SAVED_MODEL_SCHEMA: ("SAVED MODEL PATH", self.from_path),
      self.SEQUENTIAL_MODEL_SCHEMA: ("KERAS SEQUENTIAL", self.from_keras_sequential),
      self.MULTI_SEQUENTIAL_MODEL_SCHEMA: ("MULTI SEQUENTIAL", self.from_multi_sequential)
    }

    #self.MODEL_CONFIG_SCHEMA = reduce(lambda schema1, schema2: schema1 | schema2, list(self.schema_constructors.keys()))

  def from_yaml(self, yaml: yml.YAML):
    """
    Instantiate a ML model from a YAML object that contains the model's specification
    
    Args:
        yaml (yml.YAML): YAML object
    
    Returns:
        keras.Model: instantiated model
    
    Raises:
        Exception: If no matching schema is found.
    
    """
    for schema in self.schema_constructors:
      name, constructor = self.schema_constructors[schema]
      try:
        model_yaml = yml.load(yaml.as_yaml(), schema)
        logger.info(f"Model schema matched to: {name}")
        model = constructor(model_yaml)
        return model
      except Exception as e:
        logger.debug(f"Model schema did not match {name}; {e}\n Full trace: {traceback.format_exc()}")
        #print(traceback.format_exc())
    raise Exception("No valid schema matched provided model YAML; look at DEBUG log level for more info.")

  def from_path(self,model_yaml: yml.YAML):
    """
    Loads a saved model 
            
    Args:
        model_yaml (yml.YAML): YAML object
    
    Returns:
        keras.Model: instantiated model
    
    """
    if "custom_class" in model_yaml:
      model_class = model_yaml.get("custom_class").text
      return getattr(custom_models, model_class).load(model_yaml.get("path").text)
    else:
      return keras.models.load_model(model_yaml.get("path").text)

  def from_keras_sequential(self, model_yaml: yml.YAML):
    """
    Instantiate a ML model of Keras' Sequential class
    
    Args:
        model_yaml (yml.YAML): YAML object

    Returns:

        keras.Model: an instance of class Model

    """
    model_layers = model_yaml.get("kwargs").get("layers")
    layer_instances = [self.yaml_to_obj.get_instance(layer_yaml, layers) for layer_yaml in model_layers]
    return keras.Sequential(layer_instances)

  def from_multi_sequential(self, model_yaml: yml.YAML):
    """
    Instantiate a ML model that uses multiple Keras Sequential models internally
    
    Args:
        model_yaml (yml.YAML): YAML object

    Returns:

        keras.Model: an instance of class Model

    """
    args = model_yaml.get("kwargs")
    materialised_kwargs = copy.deepcopy(args.data)
    for arg in args:
      try:
        yml.load(args.get(arg).as_yaml(), self.SEQUENTIAL_MODEL_SCHEMA)
        materialised_kwargs[arg] = self.from_keras_sequential(args.get(arg))
      except Exception as e:
        logger.debug(f"Parameter {arg} does not match Sequential model schema. Full trace: {traceback.format_exc()}")
    return getattr(custom_models, model_yaml.get("class").text)(**materialised_kwargs)


class CallbackFactory(object):

  """Factory class for instantiating Tensorflow callback objects
  """
  
  @classmethod
  def create(cls, yaml: yml.YAML) -> tf_callbacks.Callback:
    """Create callback from YAML object
    
    Args:
        yaml (yml.YAML): Input YAML object
    
    Raises:
        ValueError: When YAML does not match any known callback class
    """
    yaml_to_obj = SimpleClassInstantiator()

    for module in [tf_callbacks, custom_callbacks]:
      try:
        callback = yaml_to_obj.get_instance(yaml, scope=module)
        return callback
      except AttributeError as e:
        logging.debug(f"error loading callback from YAML {yaml.dict} from module {module}: {e}\n Full trace: {traceback.format_exc()}")
    raise ValueError("Provided YAML did not match any known callback.")


class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for training configuration processing logic.

  """

  def other_setup(self):
    self.model_factory = ModelFactory()

  def get_config_schema(self):

    self.DATA_PREPROCESSING_SCHEMA = yml.Seq(self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA) | yml.EmptyList()

    self.TRAINING_CONFIG_SCHEMA = yml.Map({
      "batch_size": yml.Int(),
      "epochs": yml.Int(),
      yml.Optional("seed", default = 1): yml.Int(),
      yml.Optional(
        "metrics", 
        default = None): yml.Seq(yml.Str()) | yml.EmptyNone(),
      "loss": self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA,
      yml.Optional(
        "steps_per_epoch", 
        default = 10000): yml.Int(),
      yml.Optional(
        "optimizer", 
        default = {"class": "Adam", "kwargs": {}}): self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA,
      yml.Optional(
        "callbacks", 
        default = None): yml.Seq(self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)| yml.EmptyNone(),
      yml.Optional(
        "filters",
        default = []): self.DATA_PREPROCESSING_SCHEMA
    })

    schema = yml.Map({
      yml.Optional("input_path", default = None): self.IO_CONFIG_SCHEMA, 
      yml.Optional("output_path", default = None): self.IO_CONFIG_SCHEMA,
      "model_path": self.IO_CONFIG_SCHEMA,
      "training": self.TRAINING_CONFIG_SCHEMA,
      "model": yml.Any(),
      yml.Optional("charset", default = "lowercase"): yml.Str()
       })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.
        
    Args:
        config (yml.YAML): YAML object
    
    Returns:
        Config: Instantiated configuration for a Predictor ML stage
    
    """
    input_path, output_path = config.get("input_path").text, config.get("output_path").text
    charset = config.get("charset").text

    model_path = config.get("model_path").text
    training_config = TrainingConfig.from_yaml(config.get("training"))
    set_seed(training_config.seed)
    model = self.model_factory.from_yaml(config.get("model"))

    return Config(
      input_path = input_path, 
      output_path = output_path,
      model_path = model_path,
      model = model,
      training_config = training_config,
      charset = charset,
      yaml = config)
