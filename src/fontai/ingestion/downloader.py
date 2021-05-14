import typing
from pathlib import Path
import zipfile
import io
import sys
import logging

from PIL import ImageFont

from fontai.core import DataPath, InMemoryZipFile
from fontai.config.ingestion import Config
from fontai.ingestion.scrappers import InMemoryFile, FileScrapper

logger = logging.getLogger(__name__)

class FileBundler(object):
  """
  persists a list of files as a sequence of zip files, each with a maximum allowed pre-compression size

  folder: Path object pointing to the path in which to save the zip files

  size_limit: presize limit in MB

  kwargs: arguments passed to get_all_files() method from scrapper

  """


  def __init__(self, output_path: DataPath, size_limit: float = 128.0):

    self.chunk_size_limit = size_limit
    self.output_path = output_path

    self.chunk_id = 0

    self.bundle = InMemoryZipFile()

  def add_file(self, file: InMemoryFile)-> None:

    """
    Add a file to the list

    file: InMemoryFile object

    """
    file_size = sys.getsizeof(file.content)
    if self.bundle.size + file_size > 1e6 * self.chunk_size_limit:
      self.renew_bundle()

    self.bundle.add_file(file)

  def renew_bundle(self):
    """
    Persist current in-memory zip file bundle and open a new empty one

    """
    self.bundle.compress()
    if self.bundle.n_files > 0:
      bytestream = self.bundle.get_bytes()
      (self.output_path / str(self.chunk_id)).write_bytes(bytestream)
    self.bundle.close()
    self.chunk_id += 1
    self.bundle = InMemoryZipFile()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.renew_bundle()
    self.bundle.compress().close()


class Ingestor(object):
  """
  Ingestion pipeline, takes as arguments a configuration object that defines its execution

  config: A Config instance

  """

  def __init__(self,config: Config):

    self.config = config

  def is_fontfile(self, file: InMemoryFile) -> bool:

    try:
      ImageFont.truetype(io.BytesIO(file.content),50)
      return True
    except Exception as e:
      logger.exception(f"Error while parsing font file {file.name}")
      return False

  def run(self):

    """
      Run ingestion pipeline

    """

    with FileBundler(output_path = self.config.output_path, size_limit = self.config.max_zip_size) as bundler:
      for scrapper in self.config.scrappers:
        logger.info(f"Processing scrapper of type {scrapper.__class__.__name__}")
        for file in scrapper.get_files():
          if self.is_fontfile(file):
            bundler.add_file(file)


