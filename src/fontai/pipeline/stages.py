import logging
from pathlib import Path
import typing
import zipfile
import io
import sys
import typing as t
import pickle

from PIL import ImageFont

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from fontai.config.preprocessing import Config as ProcessingConfig, ConfigHandler as ProcessingConfigHandler
from fontai.preprocessing.file_preprocessing import PipelineExecutor, OneToManyMapper, KeyValueMapper, FontFileToLabeledExamples, FeatureCropper, FeatureResizer, BytestreamPathReader, ZipToFontFiles, TfrRecordWriter

from fontai.pipeline.base import MLPipelineTransform
from fontai.io.storage import BytestreamPath
from fontai.config.ingestion import Config as IngestionConfig, ConfigHandler as IngestionConfigHandler

from fontai.config.training import Config as TrainingConfig, ConfigHandler as TrainingConfigHandler
from fontai.training.file_preprocessing import InputPreprocessor

logger = logging.Logger(__name__)
  

class FontIngestion(ConfigurableTransform, IdentityTransform):

  """Ingestor class that retrieves zipped font files; it is initialised from a configuration object that defines its execution. It's transform method takes a list of scrappers from which it downloads files to storage.
  """
  input_file_format = InMemoryFile
  output_file_format = InMemoryFile

  def __init__(self):
    """
    """

  @classmethod
  def get_config_parse(cls):
    return IngestionConfigHandler()

  @classmethod
  def from_config(cls, config: IngestionConfig):
    return cls()

class LabeledExampleExtractor(MLPipelineTransform):
  """
  File preprocessing pipeline that maps zipped font files to Tensorflow records for ML consumption; takes a Config object that defines its execution.

  config: A Config instance

  """

  input_file_format = InMemoryZipHolder
  output_file_format = TFRecordDatasetWrapper

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
        mapper = InputToFontFiles(input_file_format = input_file_format)
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

  def transform_batch(self, reader: BatchReader, writer: BatchWriter):

    """
      Runs Beam preprocessing pipeline as defined in the config object.
    
    """

    # if output is locally persisted, create parent folders
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
          mapper = KeyValueMapper(
            mapper = self.pipeline
          )
        )
      )
      | "write to storage" >> beam.ParDo(Writer(writer)))

  @classmethod
  def get_config_parser(cls):
    return ProcessingConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "preprocessing"




class Model(FittableMLPipelineTransform):
  """
      Base class for ML pipeline stages

      config: Configuration object inheriting from BaseConfig

    """

  input_file_format = TFRecordDatasetWrapper
  output_file_format = TFRecordDatasetWrapper

  def __init__(self, config: TrainingConfig):

    self.config = config
    self.data_fetcher = InputPreprocessor()

  def __init__(self,
    model: TrainableModel,
    batch_size: int,
    n_epochs: int,
    steps_per_epoch: int):


    self.model = model
    self.optimizer = optimizer
    

  def fit_from_config(self):
    """
      Run batch processing from initial configutation

    """

    data_fetcher = InputPreprocessor()
    model = self.config.model 
    model.fit(
      data=data_fetcher.fetch_tfr_files(self.config.input_path.list_files()), 
      steps_per_epoch = args.steps_per_epoch, 
      epochs = args.n_epochs, 
      callbacks=callbacks)

    return model

  def process(self, input_data: Path) -> t.Generator[t.Any, None, None]:
    """
    Process a single instance
    """

    raise NotImplementError("This class does not have an implementation for the process() method; for scoring, use ModelScoringStage instead.")

  @classmethod
  def get_config_parser(cls):
    return TrainingConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "training"