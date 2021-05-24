from __future__ import annotations
from pathlib import Path
import io
import zipfile
import sys
import re
import typing as t
from abc import ABC, abstractmethod
import logging


from apache_beam.io.gcp.gcsio import GcsIO

from numpy import ndarray


from fontai.core.base import InMemoryFile, InMemoryZipFile

logger = logging.getLogger(__name__)



class FileHandler(ABC):
  # Interface definition for file handlers
  @abstractmethod
  def read(self, path: str) -> bytes:
    pass

  @abstractmethod
  def write(self, path: str, content: bytes) -> None:
    pass

  @abstractmethod
  def list_files(self, path: str) -> t.List[str]:
    pass


class LocalFileHandler(FileHandler):
  # Class handler for files in local storage
  def read(self, path: str) -> bytes:
    path = Path(path)
    if path.is_file():
      #return InMemoryFile(name=path.name, content=path.read_bytes())
      return path.read_bytes()
    else:
      raise Exception(f"Path ({str(path)}) does not point to file")

  def write(self, path: str, content: bytes) -> None:
    path = Path(path)
    Path(path.parent).mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

  def list_files(self, path: str) -> t.List[str]:
    path = Path(path)
    contents = path.iterdir()
    files = [str(content) for content in contents if content.is_file()]
    return files


class GcsFileHandler(FileHandler):
  # Class handler for files in Google Cloud Storage

  def read(self, url: str) -> bytes:
    #url = self.as_str(url)
    #return InMemoryFile(content=io.BytesIO(GcsIO().open(url,mode="r").read()),name=Path(url).name)
    gcs_file = GcsIO().open(url,mode="r")
    content = gcs_file.read()
    gcs_file.close()
    return content

  def write(self, url: str, content: bytes) -> None:
    #url = self.as_str(url)
    gcs_file = GcsIO().open(url,mode="w")
    gcs_file.write(content)
    gcs_file.close()

  def list_files(self,url: str) -> t.List[str]:
    #url = self.as_str(url) 
    raw_list = list(GcsIO().list_prefix(url).keys())

    return [elem for elem in raw_list if Path(elem.replace("gs://","")) != Path(url.replace("gs://",""))]


class UrlFileHandler(FileHandler):
  # Class handler for downloadable urls 

  def read(self, url: str) ->bytes:
    r = requests.get(url, stream=True)
    bf = io.BytesIO()
    with io.BytesIO() as bf:
      for chunk in r.iter_content(chunk_size=chunk_size):
        bf.write(chunk)
      content = bf.getvalue()
    return content

  def write(self, url: str, content: bytes):
    raise NotImplementedError("Files cannot be written to a url address")

  def list_files(self,url: str) -> t.List[str]:
    raise NotImplementedError("Files cannot be listed from a url address")

  def 


class FileHandlerFactory(object):
  # Factory method that determines the appropriate file handler class based on the output string path
  @classmethod
  def create(cls, path: str):

    self.PREFIX_TO_HANDLER = {
    "gs://": GcsFileHandler()
    "https://": UrlFileHandler(),
    "http://": UrlFileHandler()
    }

    for prefix in self.PREFIX_TO_HANDLER:
      if prefix in path:
        return self.PREFIX_TO_HANDLER[prefix]
    raise ValueError("No allowed prefix was detected in target path.")


class DataPath(object):
  """
    Data reader/writer class that abstracts the underlying storage location. Supports local and GCS storage and downloadable urls

    source_str: path or url pointing to the data, or to where it will be saved

  """

  def __init__(self, source_str: str):

    self.string = str(source_str)

    #extensions
    self.is_gcs = "gs://" in self.string
    self.is_http = "http://" in self.string
    self.is_https = "https://" in self.string

    self.handler = FileHandlerFactory.create(self.string)
    self.filename = self.get_filename()

  def is_url(self):
    return self.is_gcs or self.is_http or self.is_https

  def extend_url_path(self, suffix):
    """
      Returns a DataPath instance pointing to a sub files and sub folders
    """

    def extend(preffix, string, suffix):
      suffixed = string.replace(preffix,"") + "/" + suffix
      suffixed = re.sub("/+","/",suffixed)
      return DataPath(preffix + suffixed)

    if self.is_gcs:
      return extend("gs://", self.string, suffix)
    elif self.is_http:
      return extend("http://", self.string, suffix)
    elif self.is_https:
      return extend("https://", self.string, suffix)
    else:
      raise ValueError("url does not match any valid preffix.")

  def read_bytes(self) -> bytes:
    """
      Read the bystream from the path

      Returns the file's bytestream
    """
    return self.handler.read(self.string)

  def write_bytes(self,content: bytes) -> None:
    """
      Writes a bytestream to the path
    """

    self.handler.write(self.string, content)

  def list_files(self) -> t.Generator[DataPath, None, None]:
    """
      List files (but not dirs) in the folder given by source_str

      Returns a list of DataPath objects corresponding to the full path of each file.
    """

    for elem in self.handler.list_files(self.string):
      yield DataPath(elem)

  def __truediv__(self, path: str) -> DataPath:
    if not isinstance(path, str):
      raise TypeError("path must be a string")
    elif self.is_url():
      return self.extend_url_path(path)
    else:
      return DataPath(str(Path(self.string) / path))

  def __str__(self):
    return self.string

  def get_filename(self):
    if self.is_url():
      filename = self.string.split("/")[-1]
    else:
      filename = Path(self.string).name
    if filename == "":
      raise ValueError("Path does not point to a file")
    return filename




class BatchWriter(ABC):

  def __init__(self, output_path: DataPath):

  @abstractmethod
  def add_file(self, file: InMemoryFile):
    pass

  @abstractmethod
  def __enter__(self):
    pass

  @abstractmethod
  def __exit__(self):
    pass




class ZipBatcher(BatchWriter):
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
      logger.info(f"Persisting {self.bundle.n_files} files in zip file with id {self.chunk_id} ({self.bundle.size/1e6} MB)")
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

    


class LabeledExampleTfrWriter(BatchWriter):
  """
    Takes instances of LabeledExamples and writes them to a tensorflow record file.

    output_path: output destination

  """
  def __init__(self, output_path: DataPath):
    self.tfr_formatter = TfrHandler()
    self.writer = tf.io.TFRecordWriter(str(output_path))

  def img_to_png_bytes(self, img):
    bf = io.BytesIO()
    imageio.imwrite(bf,img,"png")
    val = bf.getvalue()
    bf.close()
    return val
    
  def format_tfr_contents(self, example: LabeledExample) -> t.Tuple[bytes,bytes,bytes]:
    png = self.img_to_png_bytes(example.x)
    label = str.encode(example.y)
    metadata = str.encode(example.metadata)
    return png, label, metadata

  def add_file(self, example: LabeledExample):
    tf_example = self.tfr_formatter.as_tfr(*self.format_tfr_contents(example))
    self.writer.write(tf_example.SerializeToString())

  def __enter__(self):
    return self

  def __exit__(self):
    self.writer.close()





# class FileBundler(object):
#   """
#   persists a list of files as a sequence of zip files, each with a maximum allowed pre-compression size

#   folder: Path object pointing to the path in which to save the zip files

#   size_limit: presize limit in MB

#   kwargs: arguments passed to get_all_files() method from scrapper

#   """


#   def __init__(self, output_path: DataPath, size_limit: float = 128.0):

#     self.chunk_size_limit = size_limit
#     self.output_path = output_path

#     self.chunk_id = 0

#     self.bundle = InMemoryZipFile()

#   def add_file(self, file: InMemoryFile)-> None:

#     """
#     Add a file to the list

#     file: InMemoryFile object

#     """
#     file_size = sys.getsizeof(file.content)
#     if self.bundle.size + file_size > 1e6 * self.chunk_size_limit:
#       self.renew_bundle()

#     self.bundle.add_file(file)

#   def renew_bundle(self):
#     """
#     Persist current in-memory zip file bundle and open a new empty one

#     """
#     self.bundle.compress()
#     if self.bundle.n_files > 0:
#       logger.info(f"Persisting {self.bundle.n_files} files in zip file with id {self.chunk_id} ({self.bundle.size/1e6} MB)")
#       bytestream = self.bundle.get_bytes()
#       (self.output_path / str(self.chunk_id)).write_bytes(bytestream)
#     self.bundle.close()
#     self.chunk_id += 1
#     self.bundle = InMemoryZipFile()

#   def __enter__(self):
#     return self

#   def __exit__(self, exc_type, exc_val, exc_tb):
#     self.renew_bundle()
#     self.bundle.compress().close()