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

from fontai.core.base import MLPipelineTransform, BatchWritingStage
from fontai.core.io import BytestreamPath, FileBatcher
from fontai.config.ingestion import Config as IngestionConfig, ConfigHandler as IngestionConfigHandler

from fontai.config.training import Config as TrainingConfig, ConfigHandler as TrainingConfigHandler
from fontai.training.file_preprocessing import InputPreprocessor

logger = logging.Logger(__name__)
  

class ConfigurableTransform(ABC):

  """Interface for configurable tranformations; they can be instantiated and run from YAML configuration files.
  """

  @classmethod
  @abstractmethod
  def from_config(cls, config: BaseConfig):
    """Instantiate class from a configuration object
    
    Args:
        config (BaseConfig): Config object parsed from a YAML file
    """
    pass

  @classmethod
  def parse_config(cls, path: str) -> BaseConfig:
    """Parse a YAML configuration file and create an instance inheriting from BaseConfig
    
    Args:
        path (str): Path to the YAML configuration file
    
    Returns:
        BaseConfig: Instantiated Config instance
    """
    return self.get_config_parser().from_file(BytestreamPath(path))

  @classmethod
  def from_config_file(cls, path: str):
    """Create a ConfigurableTransform instance from a YAML configuration file
    
    Args:
        path (str): Path to the YAML configuration file
    
    Returns:
        ConfigurableTransform: Instance created from configuration file.
    """
    return cls.from_config(cls.parse_config(path))

  @classmethod
  @abstractmethod
  def run_from_config_file(cls, path: str) -> None:
    """Instantiate a ConfigurableTransform instance and process a file batch from storage, saving outputs to storage, using storage locations specified in the YAML configuration file.
    
    Args:
        path (str): Path to the YAML configuration file
    """
    pass

  @classmethod
  @abstractmethod
  def get_config_parser(cls) -> BaseConfigHandler:

    """
    Returns an instance of the transform's configuration parser class
    
    """
    pass


class MLPipelineTransform(ConfigurableTransform, ABC):
  """
    Interface implemented by runnable ML stage objects. They are initialised by passing an execution configuration object and can perform batch transforming according to it; they can also do stream transforming using the `transform` method.

    config: Execution configuration object
  """

  @classmethod
  def run_from_config_file(cls, path) -> None:

    config = cls.parse_config(path)
    if config.reader is None or config.writer is None:
      raise TypeError("Configuration object does not specify reader and writer instances.")
    cls.from_config(config).transform_batch(config.reader,config.writer)

  @abstractmethod
  def transform(self, data: t.Any) -> t.Generator[t.Any]:
    """
    transforme a single input instance
    
    Args:
        data (t.Any): Input data
    
    """
    pass


    @abstractmethod
  def transform_batch(self, reader: BatchReader, writer: BatchWriter) -> None:
    """
    transformes a batch of files and persist output
    
    Args:
        reader (BatchReader): Object inheriting from BatchReader
        writer (BatchWriter): Object inheriting from BatchWriter
    
    """
    pass


  @classmethod
  @abstractmethod
  def get_stage_name(cls) -> str:
    """
    Returns a string with the stage name
    
    """
    pass

  def save(self, output_folder: str) -> None:
    """
    Persists necessary data from which the instance can be loaded again.
    
    output_folder: BytestreamPath instance pointing to output folder.
    
    Args:
        output_folder (str): path to output folder in storage media
    
    """
    output_folder = BytestreamPath(output_folder)

    logger.info(f"{self.__class__.__name__} configuration persisted at {output_folder}")
    (output_folder / "config.yaml").write_bytes(bytes(self.config.yaml.as_yaml().encode("utf-8")))

  @classmethod
  def load(cls, input_folder: str) -> MLPipelineTransform:
    """
    Load an instance of this class
        
    Args:
        input_folder (str): Path to input folder
    
    Returns:
        MLPipelineTransform: Loaded instance
    
    """
    logger.info(f"{self.__class__.__name__} loaded from {input_folder}")
    return cls.get_config_parser().from_file(input_folder / "config.yaml")    



class FittableMLPipelineTransform(MLPipelineTransform,ABC):

  """Interface for pipeline transforms that can be fitted. Scoring is done using the 'transform' method.
  """

  def fit(self, data: t.Any) -> FittableMLPipelineTransform:
    """
    Fits the stage to the passed data

    """
    pass

  def save(self, output_folder: str) -> None:
    super().save(output_path)
    self.model.save(output_path)

  def load(self, input_folder: str) -> FittableMLPipelineTransform:
    loaded = super().load(input_folder)
    loaded.model = Model.load(input_folder)
    return loaded


class FontIngestion(ConfigurableTransform):

  """Ingestor class that retrieves zipped font files; it is initialised from a configuration object that defines its execution.
  """

  def __init__(self, config: Config):
    """
    
    Args:
        config (Config): Configuration object
    """
    config = config

  def run_from_config_file(self) -> None:

    writes_counter = 0
    for scrapper in self.config.scrappers:
      for url in scrapper.get_source_urls():
        try:
          logger.debug(f"Retrieving data from {url}")
          content = url.read_bytes()
          output_path = (self.config.output_path / str(writes_counter))
          logger.debug(f"persisting to {output_path}")
          output_path.write_bytes(content)
          writes_counter += 1
        except Exception as e:
          logger.exception(f"An error ocurred when scrapping {url}: {e}")

  @classmethod
  def get_config_parse(cls):
    return IngestionConfigHandler()

  @classmethod
  def from_config(cls, config: IngestionConfig):
    



class LabeledExampleExtractor(MLPipelineTransform):
  """
  File preprocessing pipeline that maps zipped font files to Tensorflow records for ML consumption; takes a Config object that defines its execution.

  config: A Config instance

  """

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
        mapper = ZipToFontFiles()
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
  def from_config(cls, config):

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

  def __init__(self, config: TrainingConfig):

    self.config = config
    self.data_fetcher = InputPreprocessor()

  def run_from_config(self):
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