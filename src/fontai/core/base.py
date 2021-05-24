from __future__ import annotations
from pathlib import Path
import io
import zipfile
import sys
import re
import typing as t
import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel
from apache_beam.io.gcp.gcsio import GcsIO

from tensorflow import string as tf_str
from tensorflow.train import (Example as TFExample, Feature as TFFeature, Features as TFFeatures, BytesList as TFBytesList)
from tensorflow.io import FixedLenFeature, parse_single_example

from fontai.core.io import DataPath
from numpy import ndarray

logger = logging.getLogger(__name__)

class TfrHandler(object):
  """
    Class to handle tensorflow records at preprocessing stages
  """
  def __init__(self):
    self.SCHEMA = {
    'label': FixedLenFeature([], tf_str),
    'metadata': FixedLenFeature([], tf_str),
    'png': FixedLenFeature([], tf_str),
    }

  def as_tfr(self, png: bytes, label: str, metadata: str) -> TFExample:
    """
      Wraps the arguments into a TensorFlow Example instance
    """
    def bytes_feature(value):
      return TFFeature(bytes_list=TFBytesList(value=[value]))

    return TFExample(
      features=TFFeatures(
        feature={
        "png": bytes_feature(png),
        "label":bytes_feature(bytes(label)),
        "metadata":bytes_feature(bytes(metadata))}))

  def from_tfr(self, serialised):
    """
      unpacks a serialised tf record
    """
    return parse_single_example(serialised,self.SCHEMA)



class InMemoryFile(BaseModel):
  # wrapper that holds the bytestream and name of a file
  filename: str
  content: bytes

  def __eq__(self,other):
    return isinstance(other, InMemoryFile) and self.filename == other.filename and self.content == other.content 

class LabeledExample(BaseModel):
  # wrapper that holds a labeled ML example, with asociated metadata
  x: ndarray
  y: str
  metadata: str

  def __iter__(self):
    return iter((self.x,self.y,self.metadata))

  def __eq__(self,other):
    return isinstance(other, LabeledExample) and self.x == other.x and self.y == other.y and self.metadata == other.metadata

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True



class KeyValuePair(BaseModel):
  # wrapper that holds a key value pair with key of type str
  key: str
  value: t.Union[InMemoryFile, LabeledExample]

  def __iter__(self):
    return iter((self.key,self.value))

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

  def add_file(self,file: InMemoryFile):
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




class MLPipelineStage(ABC):
  """
    Interface implemented by runnable ML stage objects. They are initialised by passing an execution configuration object and can perform batch transforming according to it; they can also do stream transforming using the `transform` method.

    config: Execution configuration object
  """

  def __init__(self, config: BaseConfigHandler):

    self.config = config


  def run_from_config(self) -> None:
    if self.config.input_path is None or self.config.input_path is None:
      raise ValueError("Read or write paths weren't set in the configuration object.")
    self.transform_batch(self.config.input_path, self.config.output_path)

  @abstractmethod
  def transform(self, data: t.Generator[t.Any]) -> t.Generator[t.Any]:
    """
    transformes a single input instance

    """
    pass

  @abstractmethod
  def transform_batch(self, input_path: DataPath, output_path: DataPath) -> None:
    """
    transformes a batch of files and persist output

    """
    pass

  @classmethod
  @abstractmethod
  def get_config_parser(cls) -> BaseConfigHandler:
    """
    Returns an instance of the stage's configuration parser class

    """
    pass

  @classmethod
  @abstractmethod
  def get_stage_name(cls) -> str:
    """
    Returns a string with the stage name

    """
    pass

  def save(self, output_folder: DataPath):
    """
    Persists necessary data from which the instance can be loaded again.

    output_folder: DataPath instance pointing to output folder.

    """
    logger.info(f"{self.__class__.__name__} configuration persisted at {output_folder}")
    (output_folder / "config.yaml").write_bytes(bytes(self.config.yaml.as_yaml().encode("utf-8")))

  @classmethod
  def load(cls, input_folder: DataPath) -> MLPipelineStage:
    """
    Load an instance of this class

    input_path: DataPath instance pointing to input folder.

    """
    logger.info(f"{self.__class__.__name__} loaded from {input_folder}")
    return cls.get_config_parser().from_file(input_folder / "config.yaml")    



class StatefulStage(MLPipelineStage, ABC):

  def __init__(self, config: BaseConfigHandler):

    super().__init__(config)
    self.state: t.Any = None


  def save(self, output_folder: DataPath):
    super().save(output_path)
    self.state.save(output_path)

  @abstractmethod
  def load_state(self, input_path: DataPath):
    pass

  def load(self, input_folder):
    super().load(input_folder)
    self.load_state(input_folder)



class FittableStage(StatefulStage,ABC):

  def fit(self, data: t.Any) -> FittableStage:
    """
    Fits the stage to the passed data

    """
    self.state = self.fit_model(data)
    return self

  @abstractmethod
  def fit_model(self, data: t.Any) -> t.Any:
    pass

  @abstractmethod
  def fit_model_to_batch(self, input_path: DataPath, output_path: DataPath) -> t.Any:
    pass

  def fit_batch(self, input_path: DataPath, output_path: DataPath) -> None:
    """
    fits the stage to the passed set of files.

    """
    self.state = self.fit_model_to_batch(input_path, output_path)



class BatchWritingStage(MLPipelineStage):

  def __init__(self, config: BaseConfigHandler, writer: BatchWriter):
    super().__init__(config)
    self.writer = writer


