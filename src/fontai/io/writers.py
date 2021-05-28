"""This module contains the interfaces between ML pipelines and storage for the purpose of output writing. At the moment only zip files and Tensorflow record output files are supported; as these classes could in principle be used in distributed processing systems, output file names are randomised to avoid collitions to a reasonable degree. The main purpose of these classes is to write a sequence of output files, each with a maximum allowed size.

"""
from __future__ import annotations
from pathlib import Path
import io
import zipfile
import sys
import re
import typing as t
from abc import ABC, abstractmethod
import logging

from fontai.io.storage import BystestreamPath
from fontai.io.formats import InMemoryZipHolder, InMemoryFontfileHolder

logger = logging.getLogger(__name__)



class BatchWriter(ABC):

  """Abstract class that forms the interface of ML pipeline stages to storage media; it writes sequences of files to any storage media suported by the BytestreamPath class, in a format specified by classes inheriting from this interface.
  
  Attributes:
      output_path (str): Path to storage
      size_limit (float): Single-file size limit that is used when writing a sequence of files
  """

  def __init__(self, output_path: str, size_limit: float):

    self.output_path = output_path
    self.size_limit = size_limit

  @property
  @abstractmethod
  def writer(self) -> t.Any:
    """Instantiate file writer instance
    """
    pass

  @abstractmethod
  def add(self, file: TfrWritable) -> None:
    """Add file to the batch to be persisted
    
    Args:
        file (TfrWritable): An instance inheriting from TfrWritable
    """
    pass

  @abstractmethod
  def open(self):
    """Open a new output file 
    
    """
    pass

  @abstractmethod
  def close(self, file: t.Any):
     """Closes the current output file 
    
    """
    pass


class ZipWriter(BatchWriter):
  """
  persists a list of files as a sequence of zip files, each with a maximum allowed pre-compression size
  
  
  Attributes:
      bundle (InMemoryZipBundler): In-memory zip file to be persisted when it's closed
      shard_id (int): Written files counter
      shard_size_limit (float): maximum allowed pre-compression size for individual zip files
      output_path (BytestreamPath): Interface to storage.
  
  """

  def __init__(self, output_path: str, size_limit: float = 128.0):
    """
      
    Args:
        output_path (str): Path to the path in which to save the zip files
        size_limit (float, optional): Mximum allowed  pre-compression size in MB for individual output zip files
    """
    self.shard_size_limit = size_limit
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


  def add(self, file: InMemoryFile)-> None:

    if not isinstance(file, InMemoryFile):
      logger.info(f"Casting file to InMemoryFile instance before zipping.")
      file = file.to_in_memory_file()

    file_size = sys.getsizeof(file.content)
    if self.bundle.size + file_size > 1e6 * self.shard_size_limit:
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
    self.bundle = InMemoryZipFile()

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

    def __init__(self, output_path: str, size_limit: float = 128.0):
    self.output_path = BytestreamPath(output_path)


    #file prefix avoids file write collitions between workers
    self.file_preffix = f"{random.getrandbits(32)}-{str(datetime.datetime.now())}"
    self.shard_id = 0
    #self.writer = tf.io.TFRecordWriter(str(output_path))
    self.open()

  def shard_name(self):
    """Returns the shard name from the file preffix and written shard counter
    
    """
    return f"{self.file_preffix}-{self.shard_id}.tfr"

  def open(self, filename: str):
    self.writer = tf.io.TFRecordWriter(str(output_path / self.shard_name()))
    self.shard_size = 0
    self.shard_objs = 0

  def close(self):
    logger.info(f"Persisting {self.shard_objs} examples in tfr file with id {self.shard_id} ({self.shard_size/1e6} MB)")
    self.writer.close()
    self.shard_id += 1

  def add(self, obj: TfrWritable) -> None:
    tf_example = obj.to_tfr()
    obj_size = sys.sizeof(tf_example)

    if obj_size + self.shard_size > self.size_limit * 1e6:
      self.close()
      self.open()

    self.writer.write(tf_example.SerializeToString())
    self.shard_size += obj_size
    self.shard_objs += 1

  def __enter__(self):
    return self

  def __exit__(self):
    self.close()

