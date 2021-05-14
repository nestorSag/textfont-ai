from __future__ import annotations
from pathlib import Path
import io
import zipfile
import sys
import typing as t
from abc import ABC

from pydantic import BaseModel
from apache_beam.io.gcp.gcsio import GcsIO

from tensorflow import string as tf_str
from tensorflow.train import (Example as TFExample, Feature as TFFeature, Features as TFFeatures, BytesList as TFBytesList)
from tensorflow.io import FixedLenFeature, parse_single_example


class TfrHandler(object):
  """
    Class to handle tensorflow records in preprocessing stages
  """
  def __init__(self):
    self.SCHEMA = {
    'label': FixedLenFeature([], tf_str),
    'metadata': FixedLenFeature([], tf_str),
    'image': FixedLenFeature([], tf_str),
    }

  @classmethod
  def as_tfr(cls,png: bytes, label: str, metadata: str) -> TFExample:
    """
      Wraps the arguments into a TensorFlow Example instance
    """
    def bytes_feature(value):
      return TFFeature(bytes_list=TFBytesList(value=[value]))

    return TFExample(
      features=tf.train.TFFeatures(
        feature={
        "png": bytes_feature(png),
        "label":bytes_feature(bytes(label)),
        "metadata":bytes_feature(bytes(metadata))}))

  def from_tfr(csl, serialised):
    """
      unpacks a serialised tf record
    """
    return parse_single_example(serialised,self.SCHEMA)



class InMemoryFile(BaseModel):
  # wrapper that holds the bytestream and name of a file
  filename: str
  content: bytes



class LabeledExample(BaseModel):
  # wrapper that holds a labeled ML example, with asociated metadata
  x: object
  y: str
  metadata: str

  def __iter__(self):
    return(iter(x,y,metadata))

  def __eq__(self,other):
    return isinstance(other, LabeledExaple) and self.x == other.x and self.y == other.y and self.metadata == other.metadata

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True



class KeyValuePair(BaseModel):
  # wrapper that holds a key value pair with key of type str
  key: str
  value: object

  def __iter__(self):
    return(iter(key,value))

  def __eq__(self,other):
    return isinstance(other, KeyValuePair) and self.key == other.key and self.value == other.value

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True



class InMemoryZipFile(object):

  """
     Represents an in-memory zip file; keeps track of metadata such as total size and file count

  """
  def __init__(self):
    self.size = 0
    self.n_files = 0

    self.buffer = io.BytesIO()
    self.zip_file = zipfile.ZipFile(self.buffer,"w")

  def add_file(self,file):
    file_size = sys.getsizeof(file.content)
    self.zip_file.writestr(str(self.n_files) + file.filename, file.content)
    self.n_files += 1
    self.size += file_size

  def compress(self):
    self.zip_file.close()
    return self

  def close(self):
    self.buffer.close()

  def get_bytes(self):
    return self.buffer.getvalue()



class FileHandler(ABC):
  # Interface definition for file handlers
  @classmethod
  def read(self, path: str) -> bytes:
    pass

  @classmethod
  def write(self, path: str, content: bytes) -> None:
    pass

  @classmethod
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
  

  # def as_str(self, path):
  #   if not isinstance(path, str):
  #     return str(path)
  #   else:
  #     return path

  def read(self, url: str):
    #url = self.as_str(url)
    #return InMemoryFile(content=io.BytesIO(GcsIO().open(url,mode="r").read()),name=Path(url).name)
    gcs_file = GcsIO().open(url,mode="r")
    content = gcs_file.read()
    gcs_file.close()
    return content

  def write(self, url: str, content: bytes):
    #url = self.as_str(url)
    gcs_file = GcsIO().open(url,mode="w")
    gcs_file.write(content)
    gcs_file.close()

  def list_files(self,url: str) -> t.List[str]:
    #url = self.as_str(url) 
    raw_list = list(GcsIO().list_prefix(url).keys())

    return [elem for elem in raw_list if Path(elem.replace("gs://","")) != Path(url.replace("gs://",""))]


class FileHandlerFactory(object):
  # Factory method that determines the appropriate file handler class based on the output string path
  @classmethod
  def create(cls, path: str):
    if "gs://" in path:
      return GcsFileHandler()
    else:
      return LocalFileHandler()


class DataPath(object):
  """
    Data reader/writer class that abstracts away the underlying storage location. Supports local and GCS storage

    source_str: path or url pointing to the data, or to where it will be saved

  """

  def __init__(self, source_str: str):
    self.string = str(source_str)
    self.is_gcs = "gs://" in self.string
    self.handler = FileHandlerFactory.create(self.string)
    self.filename = self.get_filename()

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

  def list_files(self) -> t.List[DataPath]:
    """
      List files (but not dirs) in the folder given by source_str

      Returns a list of DataPath objects
    """

    return [DataPath(elem) for elem in self.handler.list_files(self.string)]

  def __truediv__(self, path: str) -> DataPath:
    if not isinstance(path, str):
      raise TypeError("path must be a string")
    elif self.is_gcs:
      return DataPath("gs://" + str(Path(self.string.replace("gs://","")) / path))
    else:
      return DataPath(str(Path(self.string) / path))

  def __str__(self):
    return self.string

  def get_filename(self):
    if self.is_gcs:
      filename = self.string.split("/")[-1]
    else:
      filename = Path(self.string).name

    if filename == "":
      raise ValueError("Path does not point to a file")
    return filename
