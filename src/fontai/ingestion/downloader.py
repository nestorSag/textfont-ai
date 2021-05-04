import typing
from pathlib import Path
import zipfile
import io
import sys

from PIL import ImageFont

from fontai.config.ingestion import *
from fontai.ingestion.retrievers import InMemoryFile, FileRetriever

logger = logging.getLogger(__name__)

class ChunkWriter(object):
  """
  persists a list of files as a sequence of zip files, each with a maximum allowed pre-compression size

  folder: Path object pointing to the folder in which to save the files

  size_limit: presize limit in MB

  kwargs: arguments passed to get_all_files() method from retriever
  """

  def __init__(self, folder: Path, size_limit: float = 128):

    self.chunk_size_limit = size_limit
    self.folder = folder

    self.chunk_size = 0
    self.n_files = 0
    self.chunk_id = 0

    self.buffer = zipfile.ZipFile(self.folder / str(self.chunk_id), "w")

  def add_file(self, file: InMemoryFile)-> None:

    """
    Add a file to the list

    file: InMemoryFile object

    """
    file_size = sys.getsizeof(file.content)
    if self.chunk_size + file_size > 1e6 * self.chunk_size_limit:
      self._close_buffer()
      self.buffer = zipfile.ZipFile(self.folder / str(self.chunk_id), "w")

    self.buffer.writestr(str(self.n_files) + file.filename, file.content)
    self.n_files += 1
    self.chunk_size += file_size

  def _close_buffer(self) -> None:
    logger.info(f"Persisting zip file: size = {self.chunk_size/1e6} MB, no. files = {self.n_files}, chunk number = {self.chunk_id}")

    self.buffer.close()

    self.chunk_size = 0
    self.n_files = 0
    self.chunk_id += 1

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._close_buffer()



class FontDownloader(object):
  """
  Download all font files and grups them into zip files

  output_folder: folder where results will be saved

  chunk_size: maximum allowed pre-compression size for resulting zip chunks
  """

  def __init__(self, output_folder: Path, chunk_size: float = 128):

    self.writer = ChunkWriter(output_folder, chunk_size)

  def download_and_compress(self, retriever: FileRetriever) -> None:
    """
    Download all font files (ttf or otf) and compress them into zip files

    retriever: Object of class FontRetriever that fetches urls and download the corresponding files

    chunk_size: precompression size in MB of each chunk

    """

    with self.writer as writer:
      for file in retriever.get_all_files():
        if self.is_fontfile(file):
          writer.add_file(file)

  def is_fontfile(self, file: InMemoryFile) -> bool:

    try:
      ImageFont.truetype(io.BytesIO(file.content),50)
      return True
    except Exception as e:
      logger.exception("Error while parsing font file")
      return False

class Ingestor(object):
  """
  Ingestion pipeline, takes as arguments a configuration object that defines its execution

  config: Either a Config object, a Path to the YAML file, or a string with the contents of the YAML configuration file

  """

  def __init__(self,config: typing.Union[Config, Path, str]) -> None:

    if isinstance(config, str):
      self.config = ConfigHandler.parse_config(config)
    elif isinstance(config,Path):
      self.config = ConfigHandler.parse_config(Path(config).read_text())
    else:
      self.config = config

  def run(self):

    """
      Run ingestion pipeline

    """

    downloader = FontDownloader(self.config.output_folder, self.config.max_zip_size)
    for retriever in self.config.retrievers:
      logger.info(f"Processing retriever of type {retriever.__class__.__name__}")
      downloader.download_and_compress(retriever = retriever)



