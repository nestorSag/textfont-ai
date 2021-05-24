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
from fontai.preprocessing.file_preprocessing import PipelineExecutor, OneToManyMapper, KeyValueMapper, FontFileToCharArrays, ArrayCropper, ArrayResizer, DataPathReader, ZipToFontFiles, TfrRecordWriter

from fontai.core.base import MLPipelineStage, BatchWritingStage
from fontai.core.io import DataPath, FileBatcher
from fontai.config.ingestion import Config as IngestionConfig, ConfigHandler as IngestionConfigHandler

from fontai.config.training import Config as TrainingConfig, ConfigHandler as TrainingConfigHandler
from fontai.training.file_preprocessing import InputPreprocessor

logger = logging.Logger(__name__)

__all__ = [
  "FileProcessingStage",
  "IngestionStage",
  "ModelTrainingStage"]
  
class FileProcessingStage(BatchWritingStage):
  """
  File preprocessing pipeline that maps zipped font files to Tensorflow records for ML consumption; takes a Config object that defines its execution.

  config: A Config instance

  """

  def __init__(self, config: ProcessingConfig, writer: FileBatcher):

    self.config = config

    self.pipeline = PipelineExecutor(
      stages = [
      OneToManyMapper(
        mapper = FontFileToCharArrays(**self.config.font_to_array_config.as_dict())
      ),
      OneToManyMapper(
        mapper = ArrayCropper()
      ),
      OneToManyMapper(
        mapper = ArrayResizer(output_size = self.config.output_array_size)
      )]
    )

  def transform(self, data):
    return self.pipeline.map(data)

  def transform_batch(self, input_path: DataPath, output_path: DataPath):

    """
      Runs Beam preprocessing pipeline as defined in the config object.
    
    """

    # if output is locally persisted, create parent folders
    if not output_path.is_gcs:
      Path(str(output_path)).mkdir(parents=True, exist_ok=True)

    pipeline_options = PipelineOptions(self.config.beam_cmd_line_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:

      input_objs_list = input_path.list_files()

      source_stream = p | beam.Create(input_objs_list)# ReadFromText(input_file_list_path)

      # execute pipeline
      (source_stream 
      | 'Load file' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = DataPathReader()
        )
      )
      | 'extract font files from zip' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = KeyValueMapper(
            mapper = ZipToFontFiles()
          )
        )
      )
      | 'get labeled exampes from font files' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = KeyValueMapper(
            mapper = self.pipeline
          )
        )
      )
      | "write to storage" >> beam.ParDo(RecordWriter(output_path, self.config.writer)))

  @classmethod
  def get_config_parser(cls):
    return ProcessingConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "preprocessing"




class IngestionStage(MLPipelineStage):
  """
  Ingestion pipeline stage, takes as arguments a configuration object that defines its execution

  config: A Config instance

  """

  def __init__(self,config: IngestionConfig):

    self.config = config

  def is_fontfile(self, file: InMemoryFile) -> bool:

    try:
      ImageFont.truetype(io.BytesIO(file.content),50)
      return True
    except Exception as e:
      logger.exception(f"Error while parsing font file {file.filename}")
      return False

  def transform_batch(self, input_path: DataPath, output_path: DataPath):

    """
      Run ingestion pipeline

    """

    with FileBatcher(output_path = output_path, size_limit = self.config.max_zip_size) as bundler:
      for zipped in input_path.list_files():
        zipped_file = InMemoryFile(filename = str(zipped), content = zipped.read_bytes())
        for subfile in self.transform(zipped_file):
          if self.is_fontfile(subfile):
            bundler.add_file(subfile)

  @classmethod
  def unpack_files_from_stream(cls, file: InMemoryFile) -> t.Generator[InMemoryFile,None,None]:
    """
    Generator method that yields all font files from a zip bytestream

    stream: bytestream from in-memory zip file

    source: name from source file, for exception logging.

    Returns tuples of the form (file bytestream, zip filename)

    """

    if source is None:
      source = "unspecified"

    def choose_ext(lst):
      ttfs = len([x for x in lst if ".ttf" in x.lower()])
      otfs = len([x for x in lst if ".otf" in x.lower()])
      if ttfs >= otfs:
        return ".ttf"
      else:
        return ".otf"

    #we assume the stream is a zip file's contents
    try:
      zipped = zipfile.ZipFile(io.BytesIO(stream))
    except Exception as e:
      logger.exception(f"Error: source ({file.filename}) can't be read as zip")
      return
    files_in_zip = zipped.namelist()
    # choose whether to proces TTFs or OTFs, but not both
    ext = choose_ext(files_in_zip)
    valid_files = sorted([filename for filename in files_in_zip if ext in filename.lower()])
    
    for file in valid_files:
      filename = Path(file).name
      try: 
        content = zipped.read(file)
        yield InMemoryFile(filename=filename, content = content)
      except Exception as e:
        logger.exception(f"Error while extracting file {file.filename} from zip")

  def transform(self, file: InMemoryFile):

    """
      Processes an in-memory bytestream from zipped font files and returns a generator of the unzipped files

      data: bytestream from zipped font files

      Returns a generator of the unzipped files.
    """
    for subfile in IngestionStage.unpack_files_from_stream(file):
      yield subfile

  @classmethod
  def get_config_parser(cls):
    return IngestionConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "ingestion"




class ModelTrainingStage(MLPipelineStage):
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