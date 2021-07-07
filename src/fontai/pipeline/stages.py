"""
  This module contains the definitions of high-level ML lifecycle stage classes; at the moment this includes ingestion, preprocessing and training/scoring.
"""

import logging
from collections import OrderedDict
from pathlib import Path
import typing
import zipfile
import traceback
import io
import os
import string
import sys
import signal
import typing as t
import pickle

from PIL import ImageFont
from numpy import ndarray
from strictyaml import as_document

from tensorflow import Tensor
import tensorflow as tf
from tensorflow.data import TFRecordDataset


import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from fontai.config.preprocessing import Config as ProcessingConfig, ConfigHandler as ProcessingConfigHandler
from fontai.config.ingestion import Config as IngestionConfig, ConfigHandler as IngestionConfigHandler
from fontai.config.prediction import ModelFactory, TrainingConfig, Config as PredictorConfig, ConfigHandler as PredictorConfigHandler


from fontai.prediction.input_processing import RecordPreprocessor
from fontai.preprocessing.mappings import PipelineFactory, BeamCompatibleWrapper, Writer

#from fontai.pipeline.base import MLPipelineTransform
from fontai.io.storage import BytestreamPath
from fontai.io.readers import ScrapperReader
from fontai.io.records import TfrWritable
from fontai.io.formats import InMemoryFile, InMemoryZipHolder
from fontai.pipeline.base import ConfigurableTransform, IdentityTransform, FittableTransform

from numpy import array as np_array

logger = logging.Logger(__name__)
  

class FontIngestion(ConfigurableTransform, IdentityTransform):

  """Ingestoion stage class that retrieves zipped font files; it is initialised from a configuration object that defines its execution. It's transform method takes a list of scrappers defined in `fontai.io.scrappers` from which it downloads files to storage.

  Configuration YAML schema:

  scrappers: list of subyamls with keys (class, kwargs) specifying the scrapper instances from `fontai.io.scrappers `to use
  output_path: target output folder
  
  """
  input_file_format = InMemoryFile
  output_file_format = InMemoryFile

  @property
  def reader_class(self):
    return ScrapperReader

  def __init__(self, config: IngestionConfig = None):
    
    self.config=config

  @classmethod
  def get_config_parser(cls):
    return IngestionConfigHandler()

  @classmethod
  def from_config_object(cls, config: IngestionConfig, **kwargs):
    return cls(config)

  @classmethod
  def run_from_config_object(cls, config: IngestionConfig):
    
    ingestor = cls.from_config_object(config)
    writer = ingestor.writer_class(ingestor.config.output_path)
    for file in ingestor.reader_class(config.scrappers).get_files():
      writer.write(file)

  @classmethod
  def get_stage_name(cls):
    return "ingestion"

  def transform_batch(self, input_path: str, output_path: str):
    raise NotImplementedError("This method is not implemented for ingestion.")


class LabeledExampleExtractor(ConfigurableTransform):
  """
  File preprocessing executable stage that maps zipped font files to Tensorflow records consisting of labeled images for ML consumption; takes a Config object that defines its execution.

  Configuration YAML schema:

  output_record_class: name of record class that will populate output files; it has to inherit from `fontai.io.records.TfrWritable`
  input_path: Input files' folder
  output_path: output files' folder
  output_array_size: height and width of the squared output images
  max_output_file_size: maximum single-file size for output files #in MB
  font_extraction_size: Font size to use when extracting characters to images
  canvas_size: initial image canvas size in which font chars are extracted
  canvas_padding: padding for the initial image canvas to allow for unusally large fonts
  beam_cmd_line_args: list of Apache Beam's command line aeguments

  """

  input_file_format = InMemoryZipHolder
  output_file_format = TFRecordDataset

  def __init__(
    self, 
    output_record_class: type,
    charset: str,
    font_extraction_size: int,
    canvas_size: int,
    canvas_padding: int,
    output_array_size: int,
    beam_cmd_line_args: t.List[str] = []):
    """
    
    Args:
        output_record_class (type): Output schema class, inheriting from TfrWritable
        charset (str): String with characters to be extracted
        font_extraction_size (int): Font size to use when extracting font images
        canvas_size (int): Image canvas size in which fonts will be extracted
        canvas_padding (int): Padding in the image extraction canvas
        output_array_size (int): Final character image size
        beam_cmd_line_args (t.List[str], optional): List of Apache Beam command line arguments for distributed processing
    """
    self.beam_cmd_line_args = beam_cmd_line_args

    self.pipeline = PipelineFactory.create(
      output_record_class = output_record_class,
      charset = charset,
      font_extraction_size = font_extraction_size,
      canvas_size = canvas_size,
      canvas_padding = canvas_padding,
      output_array_size = output_array_size)


  @classmethod
  def from_config_object(cls, config: ProcessingConfig, **kwargs):

    return cls(
      output_record_class = config.output_record_class,
      output_array_size = config.output_array_size,
      beam_cmd_line_args = config.beam_cmd_line_args,
      **config.font_to_array_config.dict())

    return cls()

  def transform(self, data):
    return self.pipeline.map(data)

  @classmethod
  def get_config_parser(cls):
    return ProcessingConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "preprocessing"

  @classmethod
  def run_from_config_object(cls, config: ProcessingConfig):

    # set provisional value for CUDA env variable to prevent out of memory errors from the GPU; this occurs because the preprocessing code depends on (CPU-bound) Tensorflow functionality, which attempts to seize memory from the GPU automatically, but this throws an error when Beam uses multiple threads.
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    output_path, input_path = config.output_path, config.input_path

    processor = cls.from_config_object(config)
    reader = processor.reader_class(input_path)

    # if output is locally persisted, create parent folders
    if not BytestreamPath(output_path).is_url():
      Path(str(output_path)).mkdir(parents=True, exist_ok=True)

    pipeline_options = PipelineOptions(processor.beam_cmd_line_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:

      # execute pipeline
      (p
      | 'create source list' >> beam.Create(reader.list_sources()) #create list of sources as strings
      | 'read files' >> beam.Map(lambda filepath: processor.input_file_format.from_bytestream_path(filepath)) #line needed to load files lazily and not overwhelm memory
      | 'get labeled exampes from zip files' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = PipelineFactory.create(
            output_record_class = config.output_record_class,
            output_array_size = config.output_array_size,
            **config.font_to_array_config.dict())
        )
      )
      | "write to storage" >> beam.ParDo(Writer(processor.writer_class(output_path, config.max_output_file_size))))

    # unset provisional value for env variable
    if visible_devices is None:
      del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
      os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices



class Predictor(FittableTransform):
  """
  This class trains a prediction model or scores new exaples with an existing prediction model.
  

  Attributes:
      model (keras.Model): Scoring model
      CHARSET_OPTIONS (t.Dict): Dictionary from allowed charsets names to charsets
      training_config (TrainingConfig): training configuration wrapper
      charset_tensor (Tensor): Tensor with an entry per character in the current charset
  

  YAML configuration schema:


  input_record_class (optional, defaults to LabeledChar): name of record class that will be parsed from input files; it has to inherit from `fontai.io.records.TfrWritable`
  input_path: Input files' folder
  output_path: output files' folder
  model_path: Output model path
  charset (optional, defaults to 'lowercase'): charset to use for training
  training: subyaml. See below
  model: subyaml. See below


  Training subyaml schema:


  batch_size (optional, defaults to None): can be null
  epochs: number of epochs
  steps_per_epoch (optional, defauls to 10k): 
  seed (optional, defaults to 1): random seed for TF functions
  metrics: List of TF metrics as strings
  loss: subyaml with keys `class` and `kwargs` to instantiate a Keras loss function
  optimizer (optional, defaults to Adam with default parameters): subyaml with keys `class` and `kwargs` to instantiate a Keras optimizer
  callbacks (optional, defaults to []): subyaml with keys `class` and `kwargs` to instantiate a Keras callback or a custome one defined in `fontai.prediction.callbacks`
  custom_filters (optional, defaults to []): subyaml with keys `name` and `kwargs` to instantiate a filtering function defined in `fontai.prediction.custom_filters` to apply just after records are deserialised for training
  custom_filters (optional, defaults to []): subyaml with keys `name` and `kwargs` to instantiate a mapping function defined in `fontai.prediction.custom_mappings` to apply just after records are deserialised for training

  model subyaml schema:

  Can be one of two:

  1. A subyaml with entries `path` and optionally `custom_class` to load an existing model from a path; `custom_class` has to be the class name if the model is custom defined in `fontai.prediction.models`

  2. A subyaml with entries `class` and `kwargs` to instantiate a Keras model architecture; currently only `Sequential` types are tested. Each Keras layers is specified and instantiated analogously in the kwargs. The class name can also correspond to a custom class in `fontai.prediction.models`. the kwargs of the specified class can subsequently be Sequential keras models if needed.



  
  """
  CHARSET_OPTIONS = {
    "uppercase": string.ascii_letters[26::],
    "lowercase": string.ascii_letters[0:26],
    "digits": string.digits,
    "all": string.ascii_letters + string.digits
    }

  input_file_format = TFRecordDataset
  output_file_format = TFRecordDataset

  def __init__(
    self, 
    model: tf.keras.Model, 
    training_config: TrainingConfig = None, 
    charset: str = "lowercase"):
    """
    
    Args:
        model (tf.keras.Model): Scoring model
        training_config (TrainingConfig, optional): Training configuration wrapper
        charset (str): charset to use for training and batch scoring. It must be one of 'lowercase', 'uppercase' or 'digits', or otherwise a string with all characters under consideration
    """
    self.model = model
    self.training_config = training_config
    self.charset = charset


    try:
      self.charset = self.CHARSET_OPTIONS[charset]
    except KeyError as e:
      logger.warning(f"Charset string is not one from {list(self.CHARSET_OPTIONS.keys())}; creating custom charset from provided string instead.")
      self.charset = "".join(list(set(charset)))

    self.num_classes = len(self.charset)
    self.charset_tensor = np_array([str.encode(x) for x in list(self.charset)])


  def fit(self, data: TFRecordDataset):
    """Fits the scoring model with the passed data
    
    Args:
        data (TFRecordDataset): training data
    
    Returns:
        Predictor: Predictor with trained model
    
    Raises:
        ValueError: If training_config is None (not provided).
    """
    if self.training_config is None:
      raise ValueError("Training configuration not provided at instantiation time.")
    
    self.model.compile(
      loss = self.training_config.loss, 
      optimizer = self.training_config.optimizer,
      metrics = self.training_config.metrics)

    self.model.fit(
      data,
      #training_data,
      steps_per_epoch = self.training_config.steps_per_epoch, 
      epochs = self.training_config.epochs,
      callbacks = self.training_config.callbacks)

    return self

  def _to_shape(self, x: t.Union[ndarray, Tensor]):
    """Reshape single example to be transformed in-memory by the `transform` method.
    
    Args:
        x (t.Union[ndarray, Tensor]): Single input
    
    Returns:
        t.Union[ndarray, Tensor]: Reshaped input
    """
    if len(x.shape) == 2:
      x = x.reshape((1,) + x.shape + (1,))
    elif len(x.shape) == 3:
      x = x.reshape((1,) + x.shape)
    return x

  def transform(self, input_data: t.Union[ndarray, Tensor, TfrWritable]) -> t.Union[Tensor, ndarray, TfrWritable]:
    """
    Process a single instance
    
    Args:
        input_data (t.Union[ndarray, Tensor, TfrWritable]): Input instance
    
    Returns:
        t.Union[Tensor, ndarray, TfrWritable]: Scored example in the corresponding format, depending on the input type.
    
    Raises:
        TypeError: If input type is not allowed.
    
    """

    if isinstance(input_data, (ndarray, Tensor)):
      return self.model.predict(self._to_shape(input_data))
    elif isinstance(input_data, TfrWritable):
      return input_data.add_score(
        score = self.model.predict(self._to_shape(input_data.features)), 
        charset_tensor=self.charset_tensor)
    else:
      raise TypeError("Input type is not one of ndarray, Tensor or TfrWritable")

  @classmethod
  def get_config_parser(cls):
    return PredictorConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "predictor"


  @classmethod
  def from_config_object(cls, config: PredictorConfig, load_from_model_path = False):
    """Initialises a Predictor instance from a configuration object
    
    Args:
        config (PredictorConfig): COnfiguration object
        load_from_model_path (bool, optional): If True, the model is loaded from the model_path argument in the configuration object.
    
    Returns:
        Predictor: Instantiated Predictor object.
    """
    if load_from_model_path:
      model_class_name = config.model.__class__.__name__
      classname_tuple = ("custom_class", model_class_name if model_class_name != "Sequential" else None)
      
      # dict -> YAML -> Model
      input_dict = {"path": config.model_path}
      if model_class_name != "Sequential":
        input_dict["custom_class"] = model_class_name
      model_yaml = as_document(input_dict)

      logger.info(f"load_from_model_path flag set to False; loading model from model_path {config.model_path} of class {model_class_name}")
      model = ModelFactory().from_yaml(model_yaml)
    else:
      model = config.model

    predictor = Predictor(model = model, training_config = config.training_config, charset = config.charset)
    return predictor

  @classmethod
  def run_from_config_object(cls, config: PredictorConfig, load_from_model_path = False):

    predictor = cls.from_config_object(config, load_from_model_path)

    data_fetcher = RecordPreprocessor(
      input_record_class = config.input_record_class,
      charset_tensor = predictor.charset_tensor,
      custom_filters = predictor.training_config.custom_filters,
      custom_mappers = predictor.training_config.custom_mappers)

    writer = predictor.writer_class(config.output_path)

    files = predictor.reader_class(config.input_path).get_files()
    data = data_fetcher.fetch(files, training_format=False, batch_size = predictor.training_config.batch_size)

    
    for features, labels, fontnames in data:
      try:
        scores = predictor.transform(features)
        
        scored_records = config.input_record_class.from_scored_batch(
          features = features.numpy(),
          labels = labels.numpy(),
          fontnames = fontnames.numpy(), 
          scores = scores,
          charset_tensor = predictor.charset_tensor)

        for record in scored_records:
          writer.write(record)
      except Exception as e:
        logger.exception(f"Exception scoring batch with features: {features}. Full trace: {traceback.format_exc()}")

  @classmethod
  def fit_from_config_object(cls, config: PredictorConfig, load_from_model_path = False):
    predictor = cls.from_config_object(config, load_from_model_path)
    
    def save_on_sigint(sig, frame):
      predictor.model.save(config.model_path)
      logger.info(f"Training stopped by SIGINT: saving current model to {config.model_path}")
      sys.exit(0)
      
    signal.signal(signal.SIGINT, save_on_sigint)

    data_fetcher = RecordPreprocessor(
      input_record_class = config.input_record_class,
      charset_tensor = predictor.charset_tensor,
      custom_filters = predictor.training_config.custom_filters,
      custom_mappers = predictor.training_config.custom_mappers)

    data = predictor.reader_class(config.input_path).get_files()

    predictor.fit(data_fetcher.fetch(data, training_format=True, batch_size=predictor.training_config.batch_size))

    logger.info(f"Saving trained model to {config.model_path}")
    predictor.model.save(config.model_path)

    return predictor

