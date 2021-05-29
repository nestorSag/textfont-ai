"""
This module provides the interfaces between ML pipeline stages and storage media for the purpose of reading input files. They relay on instances inheriting from BytestreamPath to list and retrieve available files as bytestream, that are then deserialised to specific data formats depending on the particular reader class.
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

from fontai.io.storage import BytesteamPath
from fontai.io.formats import InMemoryZipHolder, InMemoryFontfileHolder

from tf.data import TFRecordDataset

logger = logging.getLogger(__name__)

__all__ = [
  "TfrReader",
  "ZipReder"]
  
class BatchReader(ABC):

  """Interface class between ML stages and storage for file reading.
  
  Attributes:
      output_path (str): path to folder from which files are to be read
  """

  def __init__(self, input_path: str):

    self.input_path = BytestreamPath(output_path)

  @abstractmethod
  def read_files(self) -> t.Generator[t.Any]:
    """Returns a generator for the sequence of input files
    """
    pass

class TfrReader(BatchReader):

  """Class that reads a sequence of Tensorflow Record files as an instance of TFRecordDataset
  """

  def read_files(self):

    str_sources = [str(src) for src in self.input_path.list_sources()]

    return TFRecordDataset(filenames=str_sources)


class ZipReder(BatchReader):

  """Class that reads a sequence of zip files 
  """

  def read_files(self):

    for src in self.input_path.list_sources():
      yield InMemoryZipHolder(filename = str(src), content = src.read_bytes())

     
