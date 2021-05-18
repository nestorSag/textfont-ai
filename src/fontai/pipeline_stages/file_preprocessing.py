import logging
from pathlib import Path

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from fontai.config.preprocessing import Config
from fontai.preprocessing.file_preprocessing import * 

logger = logging.Logger(__name__)

class FileProcessingStage(MLPipelineStage):
  """
  File preprocessing pipeline that maps zipped font files to Tensorflow records for ML consumption; takes a Config object that defines its execution.

  config: A Config instance

  """

  def __init__(self, config: Config):

    self.config = config

    self.pipeline = PipelineExecutor(
      stages = [
      OneToManyMapper(
        mapper = ZipToFontFiles()
      ),
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

  def process(self, data):
    return self.pipeline.map(data)

  def run_from_config(self):

    """
      Runs Beam preprocessing pipeline as defined in the config object.
    
    """

    # if output is locally persisted, create parent folders
    if not self.config.output_path.is_gcs:
      Path(str(self.config.output_path)).mkdir(parents=True, exist_ok=True)

    pipeline_options = PipelineOptions(self.config.beam_cmd_line_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:

      input_objs_list = self.config.input_path.list_files()

      source_stream = p | beam.Create(input_objs_list)# ReadFromText(input_file_list_path)

      # execute pipeline
      (source_stream 
      | 'Load file' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = DataPathReader()
        )
      )
      | 'get labeled exampes from zip' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = KeyValueMapper(
            mapper = self.pipeline
          )
        )
      )
      | "write to disk" >> beam.ParDo(TfrRecordWriter(self.config.output_path)))