import logging
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

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from fontai.config.preprocessing import Config as ProcessingConfig, ConfigHandler as ProcessingConfigHandler
from fontai.config.ingestion import Config as IngestionConfig, ConfigHandler as IngestionConfigHandler
from fontai.config.training import TrainingConfig, Config as ModelConfig, ConfigHandler as ModelConfigHandler

from fontai.preprocessing.mappings import PipelineExecutor, OneToManyMapper, KeyValueMapper, FontFileToLabeledExamples, FeatureCropper, FeatureResizer, BytestreamPathReader, ZipToFontFiles, TfrRecordWriter

from fontai.pipeline.base import MLPipelineTransform
from fontai.io.readers import ScrapperReader
from fontai.io.records import ScoredLaebeledExample


logger = logging.Logger(__name__)
  

class FontIngestion(ConfigurableTransform, IdentityTransform):

  """Ingestor class that retrieves zipped font files; it is initialised from a configuration object that defines its execution. It's transform method takes a list of scrappers from which it downloads files to storage.
  """
  input_file_format = InMemoryFile
  output_file_format = InMemoryFile

  @property
  def reader_class(self):
    return ScrapperReader

  def __init__(self, config: IngestionConfig = None):
    
    self.config: = config

  @classmethod
  def get_config_parse(cls):
    return IngestionConfigHandler()

  @classmethod
  def from_config_object(cls, config: IngestionConfig):
    return cls(config)

  @classmethod
  def run_from_config_file(cls, path: str):
    
    config = cls.parse_config_file(path)
    ingestor = cls.from_config_object(config)
    writer = ingestor.writer_class(ingestor.config.output_path)
    for file in ingestor.reader_class(config.scrappers).get_files():
      writer.write(file)

  @classmethod
  def run_from_config(cls, config: IngestionConfig):
    
    ingestor = cls.from_config_object(config)
    writer = ingestor.writer_class(ingestor.config.output_path)
    for file in ingestor.reader_class(config.scrappers).get_files():
      writer.write(file)


class LabeledExampleExtractor(MLPipelineTransform):
  """
  File preprocessing pipeline that maps zipped font files to Tensorflow records for ML consumption; takes a Config object that defines its execution.

  config: A Config instance

  """

  input_file_format = InMemoryZipHolder
  output_file_format = TFDatasetWrapper

  def __init__(
    self, 
    charset: str,
    font_extraction_size: int,
    canvas_size: int,
    canvas_padding: int,
    output_array_size: int,
    beam_cmd_line_args: t.List[str] = []):

    self.beam_cmd_line_args = beam_cmd_line_args

    self.pipeline = PipelineExecutor(
      stages = [
      OneToManyMapper(
        mapper = InputToFontFiles()
      ),
      OneToManyMapper(
        mapper = FontFileToLabeledExamples(
          charset = charset,
          font_extraction_size = font_extraction_size,
          canvas_size = canvas_size,
          canvas_padding = canvas_padding)
      ),
      OneToManyMapper(
        mapper = FeatureCropper()
      ),
      OneToManyMapper(
        mapper = FeatureResizer(output_size = output_array_size)
      )]
    )


  @classmethod
  def from_config(cls, config: Config):

    return cls(
      output_array_size = config.output_array_size,
      beam_cmd_line_args = config.beam_cmd_line_args,
      **config.font_to_array_config.dict())


    return cls()
  def transform(self, data):
    return self.pipeline.map(data)

  def transform_batch(self, input_path: str, output_path: str):

    """
      Runs Beam preprocessing pipeline as defined in the config object.
    
    """

    # if output is locally persisted, create parent folders
    reader = self.reader_class(input_path)
    writer = self.writer_class(output_path)

    if not output_path.is_gcs:
      Path(str(output_path)).mkdir(parents=True, exist_ok=True)

    pipeline_options = PipelineOptions(self.config.beam_cmd_line_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:

      # execute pipeline
      (p 
      | 'Read from storage' >> beam.Create(reader.read_files()) 
      | 'get labeled exampes from zip files' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = self.pipeline
        )
      )
      | "write to storage" >> beam.ParDo(Writer(writer)))

  @classmethod
  def get_config_parser(cls):
    return ProcessingConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "preprocessing"

  @classmethod
  def run_from_config_file(cls, path: str):
    
    config = cls.parse_config_file(path)
    processor = cls.from_config_object(config)
    processor.transform_batch(input_path=config.input_path, output_path=config.output_path)





class ModelHolder(FittableMLPipelineTransform):
  """
      Base class for ML pipeline stages

      config: Configuration object inheriting from BaseConfig

    """

  input_file_format = TFDatasetWrapper
  output_file_format = TFDatasetWrapper

  def __init__(self,
    model: tf.keras.Model,
    batch_size: int,
    n_epochs: int,
    steps_per_epoch: int,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    seed = 1):


    self.model = model
    self.training_config = TrainingConfig(
      batch_size = batch_size,
      epochs = epochs,
      steps_per_epoch = steps_per_epoch,
      optimizer = optimizer,
      loss = loss,
      seed = seed)

  @classmethod
  def fit_from_config_file(cls, path: str):
    

    config = cls.parse_config_file(path)
    predictor = cls.from_config_object(config)

    data_fetcher = LabeledExamplePreprocessor(
      batch_size = predictor.training_config.batch_size,
      charset = predictor.training_config.char_set,
      filters = predictor.training_config.filters)
    
    predictor.model.compile(
      loss = predictor.training_config.loss, 
      optimizer = predictor.training_config.optimizer)

    predictor.model.fit(
      data=data_fetcher.fetch_tfr_files(config.input_path),
      steps_per_epoch = predictor.training_config.steps_per_epoch, 
      epochs = predictor.training_config.n_epochs)

    predictor.save_model(config.output_path)

  def process(self, input_data: t.Union[ndarray, Tensor, LabeledExample]) -> t.Generator[t.Any, None, None]:
    """
    Process a single instance
    """
    if isinstance(input_data, (ndarray, Tensor)):
      return self.model.predict(input_data)
    elif isinstance(input_data, LabeledExample):
      return ScoredLaebeledExample(labeled_example = input_data, score = self.model.predict(input_data.features))
    else:
      raise TypeError("Input type is not one of ndarra, Tensor or LabeledExample")

  @classmethod
  def get_config_parser(cls):
    return TrainingConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "training"