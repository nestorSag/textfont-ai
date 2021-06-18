"""This module contains the interfaces between ML pipelines and storage for the purpose of output writing. At the moment only zip files and Tensorflow record output files are supported; as these classes could in principle be used in distributed processing systems, output file names are randomised to avoid collitions to a reasonable degree. The main purpose of these classes is to write a sequence of output files, each with a maximum allowed size.

"""
from __future__ import annotations
from pathlib import Path
import io
import zipfile
import sys
import random
import re
import datetime
import typing as t
from abc import ABC, abstractmethod
import logging

from fontai.io.storage import BytestreamPath
from fontai.io.formats import InMemoryZipHolder, InMemoryFontfileHolder

from tensorflow.io import TFRecordWriter
from tensorflow.data import TFRecordDataset


logger = logging.getLogger(__name__)


__all__ = [
  "TfrWriter",
  "ZipWriter"]
  
class BatchWriter(ABC):

  """Abstract class that forms the interface of ML pipeline stages to storage media; it writes sequences of files to any storage media suported by the BytestreamPath class, in a format specified by classes inheriting from this interface.
  
  Attributes:
      output_path (str): Path to storage
      size_limit (float): Single-file size limit that is used when writing a sequence of files
  """

  def __init__(self, output_path: str, size_limit: float):

    self.output_path = output_path
    self.size_limit = size_limit

  @abstractmethod
  def write(self, obj: t.Any) -> None:
    """Add object to the file batch to be persisted
    
    Args:
        obj (t.Any): The data to be persisted
    """
    pass

  @abstractmethod
  def open(self):
    """Open a new output file 
    
    """
    pass

  @abstractmethod
  def close(self):
    """Closes the current output file 
    
    """
    pass


class ZipWriter(BatchWriter):
  """
  persists a list of files as a sequence of zip files, each with a maximum allowed pre-compression size
  
  
  Attributes:
      bundle (InMemoryZipBundler): In-memory zip file to be persisted when it's closed
      shard_id (int): Written files counter
      max_output_file_size (float): maximum allowed pre-compression size for individual zip files
      output_path (BytestreamPath): Interface to storage.
  
  """

  def __init__(self, output_path: str, max_output_file_size: float = 128.0):
    """
      
    Args:
        output_path (str): Path to the path in which to save the zip files
        max_output_file_size (float, optional): Mximum allowed  pre-compression size in MB for individual output zip files
    """
    self.max_output_file_size = max_output_file_size
    self.output_path = BytestreamPath(output_path)

    self.bundle = InMemoryZipBundler()

    self.file_preffix = f"{random.getrandbits(32)}-{str(datetime.datetime.now())}"
    self.shard_id = 0
    self.shard_size = 0
    #self.writer = tf.io.TFRecordWriter(str(output_path))
    self.open()

  def shard_name(self):
    """Returns the shard name from the file preffix and written shard counter
    
    """
    return f"{self.file_preffix}-{self.shard_id}.tfr"


  def write(self, file: InMemoryFile)-> None:

    if not isinstance(file, InMemoryFile):
      logger.info(f"Casting file to InMemoryFile instance before zipping.")
      file = file.to_in_memory_file()

    file_size = sys.getsizeof(file.content)
    if self.bundle.size + file_size > 1e6 * self.max_output_file_size:
      self.close()
      self.open()

    self.bundle.add_file(file)

  def close(self):
    self.bundle.compress()
    if self.bundle.n_files > 0:
      logger.info(f"Persisting {self.bundle.n_files} files in zip file with id {self.shard_id} ({self.bundle.size/1e6} MB)")
      bytestream = self.bundle.get_bytes()
      (self.output_path / str(self.shard_id)).write_bytes(bytestream)
    self.bundle.close()

  def open(self):
    self.shard_id += 1
    self.bundle = InMemoryZipBundler()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

    


class TfrWriter(BatchWriter):

  """Takes instances of TfrWritable data classes and writes them to a tensorflow record file according to their schema.schema.
    
    Attributes:
        file_preffix (str): random file preffix to avoid collitions with concurrent writes
        obj_size (int): Description
        output_path (BytestreamPath): interface to storage
        shard_id (int): written files counter
        shard_size (int): size of current shard
        shard_objs (int): number of objects in current shard
        writer (tf.io.TFRecordWriter): writer object
  """

  def __init__(self, output_path: str, max_output_file_size: float = 128.0):
    """
    
    Args:
        output_path (str): Output folder path
        max_output_file_size (float, optional): Maximum individual file size
    """
    
    self.output_path = BytestreamPath(output_path)
    # make sure that output directory exists
    if not BytestreamPath(self.output_path).is_url():
      Path(str(self.output_path)).mkdir(parents=True, exist_ok=True)
      
    self.max_output_file_size = max_output_file_size

    #file prefix avoids file write collitions between workers
    self.file_preffix = f"{random.getrandbits(32)}-{str(datetime.datetime.now())}"
    self.shard_id = 0

    #self.writer = tf.io.TFRecordWriter(str(output_path))
    self.writer = None
    #self.open()

  def shard_name(self):
    """Returns the shard name from the file preffix and written shard counter
    
    """
    return f"{self.file_preffix}-{self.shard_id}.tfr"

  def open(self):
    self.writer = TFRecordWriter(str(self.output_path / self.shard_name()))
    self.shard_size = 0
    self.shard_objs = 0

  def close(self):
    logger.info(f"Persisting {self.shard_objs} examples in tfr file with id {self.shard_id} ({self.shard_size/1e6} MB)")
    self.writer.close()
    self.shard_id += 1

  def write(self, obj: TfrWritable) -> None:
    if self.writer is None:
      self.open()

    tf_example = obj.to_tfr()
    obj_size = sys.getsizeof(tf_example)

    if obj_size + self.shard_size > self.max_output_file_size * 1e6:
      self.close()
      self.open()

    self.writer.write(tf_example.SerializeToString())
    self.shard_size += obj_size
    self.shard_objs += 1

  def __enter__(self):
    return self

  def __exit__(self):
    self.close()


class FileWriter(BatchWriter):

  def __init__(self, output_path: str):
    self.output_path = BytestreamPath(output_path)


    #file prefix avoids file write collitions between workers
    self.file_preffix = f"{random.getrandbits(32)}-{str(datetime.datetime.now())}"
    self.shard_id = 0


  def shard_name(self):
    """Returns the shard name from the file preffix and written shard counter
    
    """
    return f"{self.file_preffix}-{self.shard_id}"

  def write(self, file: InMemoryFile) -> None:
    x = BytestreamPath(file.filename).filename
    print(f"x: {x}, file.filename: {file.filename}, filename from Path: {Path(file.filename).name}")
    (self.output_path / x).write_bytes(file.content)
    self.shard_id += 1

  def open(self):
    """Open a new output file 
    
    """
    pass

  def close(self, file: t.Any):
    """Closes the current output file 
    
    """
    pass


class WriterClassFactory(object):
  """Factory class that returns the appropriate writer depending on the expected output file format.
  """
  @classmethod
  def get(cls, file_format: type) -> type:
    """Returns a writer type for instantiation at runtime.
    
    Args:
        file_format (InMemoryFile): Expected input file format.
    
    """
    if file_format == TFRecordDataset:
      return TfrWriter
    else:
      logger.info(f"Writer class defaulted to FileWriter; writing individual files.")
      return FileWriter