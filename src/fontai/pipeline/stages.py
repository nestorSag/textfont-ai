"""
  This module contains the definitions of high-level ML lifecycle stage classes; at the moment this includes ingestion, preprocessing and training.
"""

import logging
from collections import OrderedDict
from pathlib import Path
import typing
import zipfile
import io
import sys
import signal
import typing as t
import pickle

from PIL import ImageFont
from numpy import ndarray
from tensorflow import Tensor
from strictyaml import as_document
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
from fontai.io.records import ScoredLabeledChar, LabeledChar, TfrWritable
from fontai.io.formats import InMemoryFile, InMemoryZipHolder, TFRecordDataset
from fontai.pipeline.base import ConfigurableTransform, IdentityTransform, FittableTransform


logger = logging.Logger(__name__)
  

class FontIngestion(ConfigurableTransform, IdentityTransform):

  """Ingestoion stage class that retrieves zipped font files; it is initialised from a configuration object that defines its execution. It's transform method takes a list of scrappers from which it downloads files to storage.
  
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
  File preprocessing executable stage that maps zipped font files to Tensorflow records consisting of individual font characters for ML consumption; takes a Config object that defines its execution.

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



class Predictor(FittableTransform):
  """
  This class trains a prediction model or scores new exaples with an existing prediction model.
  

  Attributes:
      model (keras.Model): Scoring model
      training_config (TrainingConfig): training configuration wrapper
      charset (str): charset to use for training and batch scoring
  
  """

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
        charset (str): charset to use for training and batch scoring
    """
    self.model = model
    self.training_config = training_config
    self.charset = charset

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #   tf_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


  def add_batch_shape_signature(self, data: TFRecordDataset) -> TFRecordDataset:
    """Intermediate method required to make training data shapes known at graph compile time. Returns the passed data wrapped in a callable object with explicit output shape signatures
    
    Args:
        data (TFRecordDataset): Input training data
    
    Returns:
        TFRecordDataset
    
    Raises:
        ValueError
    """
    def callable_data():
      return data

    features, labels = next(iter(data))
    # drop batch size form shape tuples
    ftr_shape = features.shape[1::]
    lbl_shape = labels.shape[1::]

    if len(ftr_shape) != 3 or len(lbl_shape) != 1:
      raise ValueError(f"Input shapes don't match expected: got shapes {features.shape} and {labels.shape}")

    training_data = tf.data.Dataset.from_generator(
      callable_data, 
      output_types = (
        features.dtype, 
        labels.dtype
      ),
      output_shapes=(
        tf.TensorShape((None,) + ftr_shape),
        tf.TensorShape((None,) + lbl_shape)
      )
    )

    return training_data


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

    training_data = self.add_batch_shape_signature(data)

    self.model.fit(
      #data,
      training_data,
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
        t.Union[Tensor, ndarray, ScoredLabeledChar]: Scored example in the corresponding format, depending on the input type.
    
    Raises:
        TypeError: If input type is not allowed.
    
    """

    if isinstance(input_data, (ndarray, Tensor)):
      return self.model.predict(self._to_shape(input_data))
    elif isinstance(input_data, TfrWritable):
      return input_data.add_score(self.model.predict(self._to_shape(input_data.features)))
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
      batch_size = predictor.training_config.batch_size,
      charset = predictor.charset,
      custom_filters = [],
      custom_mappers = [])

    writer = predictor.writer_class(config.output_path)

    data = predictor.reader_class(config.input_path).get_files()
    for example in data_fetcher.fetch(data, training_format=False):
      formatted = config.input_record_class.from_parsed_bytes_dict(example)
      writer.write(predictor.transform(formatted))

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
      batch_size = predictor.training_config.batch_size,
      charset = predictor.charset,
      custom_filters = predictor.training_config.custom_filters,
      custom_mappers = predictor.training_config.custom_mappers)

    data = predictor.reader_class(config.input_path).get_files()

    predictor.fit(data_fetcher.fetch(data, training_format=True))

    logger.info(f"Saving trained model to {config.model_path}")
    predictor.model.save(config.model_path)

    return predictor

