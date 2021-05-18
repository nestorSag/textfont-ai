import typing
from pathlib import Path
import zipfile
import io
import sys
import logging
import typing as t
import pickle

from PIL import ImageFont

from fontai.core import MLPipelineStage, DataPath, InMemoryFile
from fontai.ingestion.bundler import FileBundler
from fontai.config.ingestion import Config

logger = logging.getLogger(__name__)

class IngestionStage(MLPipelineStage):
  """
  Ingestion pipeline stage, takes as arguments a configuration object that defines its execution

  config: A Config instance

  """

  def __init__(self,config: Config):

    self.config = config

  def is_fontfile(self, file: InMemoryFile) -> bool:

    try:
      ImageFont.truetype(io.BytesIO(file.content),50)
      return True
    except Exception as e:
      logger.exception(f"Error while parsing font file {file.filename}")
      return False

  def run_from_config(self):

    """
      Run ingestion pipeline

    """

    with FileBundler(output_path = self.config.output_path, size_limit = self.config.max_zip_size) as bundler:
      for scrapper in self.config.scrappers:
        logger.info(f"Processing scrapper of type {scrapper.__class__.__name__} with source {scrapper.get_source_string()}")
        for source, stream in scrapper.get_stream_tuples():
          for file in self.process(stream, source):
            if self.is_fontfile(file):
              bundler.add_file(file)

  @classmethod
  def unpack_files_from_stream(cls, stream: bytes, source: t.Optional[str] = None) -> t.Generator[InMemoryFile,None,None]:
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
      logger.exception(f"Error: source ({source}) can't be read as zip")
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
        logger.exception(f"Error while extracting file {filename} from zip")

  def process(self, data: bytes, source: str = None):

    """
      Processes an in-memory bytestream from zipped font files and returns a generator of the unzipped files

      data: bytestream from zipped font files

      source: string representation of source (i.e. url or file path)

      Returns a generator of the unzipped files.
    """
    for file in Ingestion.unpack_files_from_stream(stream=data, source=source):
      yield file