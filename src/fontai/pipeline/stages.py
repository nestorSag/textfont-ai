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


from fontai.prediction.input_processing import InputPreprocessorFactory
from fontai.preprocessing.mappings import PipelineExecutor, ManyToManyMapper, FontFileToLabeledExamples, FeatureCropper, FeatureResizer, InputToFontFiles, Writer, BeamCompatibleWrapper

#from fontai.pipeline.base import MLPipelineTransform
from fontai.io.storage import BytestreamPath
from fontai.io.readers import ScrapperReader
from fontai.io.records import ScoredLabeledExample, LabeledExample
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
  File preprocessing executable stage that maps zipped font files to Tensorflow records for ML consumption; takes a Config object that defines its execution.

  """

  input_file_format = InMemoryZipHolder
  output_file_format = TFRecordDataset

  def __init__(
    self, 
    charset: str,
    font_extraction_size: int,
    canvas_size: int,
    canvas_padding: int,
    output_array_size: int,
    beam_cmd_line_args: t.List[str] = []):
    """
    
    Args:
        charset (str): String with characters to be extracted
        font_extraction_size (int): Font size to use when extracting font images
        canvas_size (int): Image canvas size in which fonts will be extracted
        canvas_padding (int): Padding in the image extraction canvas
        output_array_size (int): Final character image size
        beam_cmd_line_args (t.List[str], optional): List of Apache Beam command line arguments for distributed processing
    """
    self.beam_cmd_line_args = beam_cmd_line_args

    self.pipeline = self.build_pipeline(
      charset = charset,
      font_extraction_size = font_extraction_size,
      canvas_size = canvas_size,
      canvas_padding = canvas_padding,
      output_array_size = output_array_size)


  def build_pipeline(
    self, 
    charset: str,
    font_extraction_size: int,
    canvas_size: int,
    canvas_padding: int,
    output_array_size: int) -> PipelineExecutor:
    """Build file processing pipeline object
    
    Args:
        charset (str): String with characters to be extracted
        font_extraction_size (int): Font size to use when extracting font images
        canvas_size (int): Image canvas size in which fonts will be extracted
        canvas_padding (int): Padding in the image extraction canvas
        output_array_size (int): Final character image size
    
    Returns:
        PipelineExecutor: Procesing transformation object
    """
    return PipelineExecutor(
      stages = [
      InputToFontFiles(),
      ManyToManyMapper(
        mapper = FontFileToLabeledExamples(
          charset = charset,
          font_extraction_size = font_extraction_size,
          canvas_size = canvas_size,
          canvas_padding = canvas_padding)
      ),
      ManyToManyMapper(
        mapper = FeatureCropper()
      ),
      ManyToManyMapper(
        mapper = FeatureResizer(output_size = output_array_size)
      )]
    )

  @classmethod
  def from_config_object(cls, config: ProcessingConfig, **kwargs):

    return cls(
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
          mapper = processor.build_pipeline(
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

  def __init__(self, model: tf.keras.Model, training_config: TrainingConfig = None, charset: str = "lowercase"):
    """
    
    Args:
        model (tf.keras.Model): Scoring model
        training_config (TrainingConfig, optional): Training configuration wrapper
        charset (str): charset to use for training and batch scoring
    """
    self.model = model
    self.training_config = training_config
    self.charset = charset
    self.input_preprocessor = InputPreprocessorFactory.create(type(model))

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #   tf_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

    data_fetcher = self.input_preprocessor(
      batch_size = self.training_config.batch_size,
      charset = self.charset,
      filters = self.training_config.filters)
    
    self.model.compile(
      loss = self.training_config.loss, 
      optimizer = self.training_config.optimizer,
      metrics = self.training_config.metrics)

    self.model.fit(
      data_fetcher.fetch(data, training_format=True),
      steps_per_epoch = self.training_config.steps_per_epoch, 
      epochs = self.training_config.epochs,
      callbacks=self.training_config.callbacks)

    return self

  def transform(self, input_data: t.Union[ndarray, Tensor, LabeledExample]) -> t.Union[Tensor, ndarray, ScoredLabeledExample]:
    """
    Process a single instance
    
    Args:
        input_data (t.Union[ndarray, Tensor, LabeledExample]): Input instance
    
    Returns:
        t.Union[Tensor, ndarray, ScoredLabeledExample]: Scored example in the corresponding format, depending on the input type.
    
    Raises:
        TypeError: If input type is not allowed.
    
    """
    if isinstance(input_data, (ndarray, Tensor)):
      return self.model.predict(input_data)
    elif isinstance(input_data, LabeledExample):
      return ScoredLabeledExample(
        labeled_example = input_data, 
        score = self.model.predict(input_data.features.reshape((1,) + input_data.features.shape + (1,)))) #images have to be reshaped to 4 dimensions to be scored
    else:
      raise TypeError("Input type is not one of ndarray, Tensor or LabeledExample")

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

    data_fetcher = predictor.input_preprocessor(
      batch_size = predictor.training_config.batch_size,
      charset = predictor.charset,
      filters = [])

    writer = predictor.writer_class(config.output_path)

    data = predictor.reader_class(config.input_path).get_files()
    for batch in data_fetcher.fetch(data, training_format=False):
      features, labels, fontnames = batch
      score = predictor.model.predict(features) #first element of example are the features

      current_batch_size = features.shape[0]
      for k in range(current_batch_size):
        writer.write(
          ScoredLabeledExample(
            labeled_example = LabeledExample(
              features = features[k].numpy(),
              label = labels[k].numpy().decode("utf-8"),
              fontname = fontnames[k].numpy().decode("utf-8")), 
            score = tf.convert_to_tensor(score[k])))

  @classmethod
  def fit_from_config_object(cls, config: PredictorConfig, load_from_model_path = False):
    
    predictor = cls.from_config_object(config, load_from_model_path)
    predictor.fit(data = predictor.reader_class(config.input_path).get_files())
    logger.info(f"Saving trained model to {config.model_path}")
    predictor.model.save(config.model_path)
    return predictor

