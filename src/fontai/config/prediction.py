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
import fontai.prediction.custom_filters as custom_filters
import fontai.prediction.custom_mappers as custom_mappers

import fontai.prediction.models as custom_models

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.callbacks as tf_callbacks
from tensorflow.random import set_seed

import fontai.prediction.callbacks as custom_callbacks
import fontai.io.records as records

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):

  """
  Training configuration wrapper for a Scoring ML stage
  
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
  custom_filters: t.List[t.Callable] = []
  custom_mappers: t.List[t.Callable] = []
  batch_size: t.Optional[PositiveInt]
  epochs: PositiveInt
  steps_per_epoch: PositiveInt
  optimizer: keras.optimizers.Optimizer
  loss: keras.losses.Loss
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
    
    if  yaml.get("custom_filters").data != []:
      args["custom_filters"] = [getattr(custom_filters, subYAML.get("name").text)(**subYAML.get("kwargs").data) for subYAML in yaml.get("custom_filters")]
    else:
      args["custom_filters"] = []


    if  yaml.get("custom_mappers").data != []:
      args["custom_mappers"] = [getattr(custom_mappers, subYAML.get("name").text)(**subYAML.get("kwargs").data) for subYAML in yaml.get("custom_mappers")]
    else:
      args["custom_mappers"] = []


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
      input_record_class (records.TfrWritable): Input schema class from the `textai.io.records` submodule
      training_config (TrainingConfig): Runtime configuration for training routine
      model (keras.Model): Model instance that's going to be trained
  
  """
  training_config: TrainingConfig
  input_record_class: type
  model_path: str
  model: keras.Model
  charset: str

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

  @validator("charset")
  def allowed_charsets(charset: str):
    """Validator for charset attribute
    
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

  @validator("input_record_class")
  def validate_input_record_class(input_record_class: type):
    """Validate input record class
    
    Args:
        input_record_class (type)
    
    Returns:
        type: Validated record class
    
    Raises:
        TypeError: If record class not in allowed set
    """
    if issubclass(input_record_class, records.TfrWritable):
      return input_record_class
    else:
      raise TypeError(f"input_record_class must inherit from TfrWritable")


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
        #print(f"Model schema did not match {name}; {e}\n Full trace: {traceback.format_exc()}")
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
        message = f"error loading callback from YAML {yaml.data} from module {module}: {e}\n Full trace: {traceback.format_exc()}"
        logging.debug(message)
    raise ValueError("Provided YAML did not match any known callback.")


class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for training configuration processing logic.

  """

  def other_setup(self):
    self.model_factory = ModelFactory()

  @classmethod
  def get_config_schema(self):
    """
    YAML configuration schema:

    input_record_class (optional, defaults to LabeledChar): name of record class that will be parsed from input files; it has to inherit from `fontai.io.records.TfrWritable`. At the moment only `LabeledChar` and `LabeledFont` are supported. If `LabeledChar`, loaded elements correspond to single images from unordered fonts. If `LabeledFont`, loaded elements correspond to tensors holding all characters corresponding to a single font.
    input_path: Input files' folder with TF record files
    output_path: output files' folder when scoring new data
    model_path: Output model path when training a model
    charset (optional, defaults to 'lowercase'): One of 'uppercase', 'lowercase', 'digits' or 'all', and determines the set to use for training or scoring.
    training: subYAML. See below
    model: subYAML. See below


    'training' subYAML schema:

    batch_size (optional, defaults to None): can be null
    epochs: number of epochs
    steps_per_epoch (optional, defauls to 10k): minibatches per epoch; this is needed because the total number of valid records is not known before runtime
    seed (optional, defaults to 1): random seed for TF functions
    metrics: List of TF metrics as strings
    loss: subYAML with keys `class` and `kwargs` to instantiate a Keras loss function
    optimizer (optional, defaults to Adam with default parameters): subYAML with keys `class` and `kwargs` to instantiate a Keras optimizer
    callbacks (optional, defaults to []): subYAML with keys `class` and `kwargs` to instantiate a Keras callback or a custom one defined in `fontai.prediction.callbacks`
    custom_filters (optional, defaults to []): subYAML with keys `name` and `kwargs` to instantiate a filtering function defined in `fontai.prediction.custom_filters` to apply just after records are deserialised for training
    custom_mappers (optional, defaults to []): subYAML with keys `name` and `kwargs` to instantiate a mapping function defined in `fontai.prediction.custom_mappings` to apply just after records are deserialised for training

    model subYAML schema:

    Can be one of two:

    1. A subYAML with entries `path` and optionally `custom_class` to load an existing model from a path; `custom_class` has to be the class name if the model is custom defined in `fontai.prediction.models`

    2. A subYAML with entries `class` and `kwargs` to instantiate a Keras model architecture; currently only `Sequential` types are tested. Each Keras layers is specified and instantiated analogously in the kwargs. The class name can also correspond to a custom class in `fontai.prediction.models`. the kwargs of the specified class can subsequently be Sequential keras models if needed.
    """

    #self.DATA_PREPROCESSING_SCHEMA = yml.Seq(self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA) | yml.EmptyList()

    self.CUSTOM_FUNCTIONS = yml.Seq(yml.Map(
        {"name": yml.Str(), 
        yml.Optional("kwargs", default = {}): yml.MapPattern(
          yml.Str(),
          self.yaml_to_obj.ANY_PRIMITIVES) | yml.EmptyDict()})) | yml.EmptyList()

    self.TRAINING_CONFIG_SCHEMA = yml.Map({
      yml.Optional("batch_size", default=None): yml.Int() | yml.EmptyNone(),
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
        "custom_filters",
        default = []): self.CUSTOM_FUNCTIONS,
      yml.Optional(
        "custom_mappers",
        default = []): self.CUSTOM_FUNCTIONS
    })

    schema = yml.Map({
      yml.Optional("input_path", default = None): self.IO_CONFIG_SCHEMA, 
      yml.Optional("output_path", default = None): self.IO_CONFIG_SCHEMA,
      yml.Optional("input_record_class",default = "LabeledChar"): yml.Str(),
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
        Config: Instantiated configuration for a Scoring ML stage
    
    """
    input_path, output_path = config.get("input_path").text, config.get("output_path").text
    charset = config.get("charset").text

    input_record_class = getattr(records, config.get("input_record_class").text)

    model_path = config.get("model_path").text
    training_config = TrainingConfig.from_yaml(config.get("training"))
    set_seed(training_config.seed)
    model = self.model_factory.from_yaml(config.get("model"))

    return Config(
      input_record_class = input_record_class,
      input_path = input_path, 
      output_path = output_path,
      model_path = model_path,
      model = model,
      training_config = training_config,
      charset = charset,
      yaml = config)
