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

from fontai.io.storage import BytestreamPath
from fontai.io.formats import InMemoryFile, TFDatasetWrapper, InMemoryZipHolder, InMemoryFontfileHolder
from fontai.io.scrappers import Scrapper

from tensorflow.data import TFRecordDataset

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

    self.input_path = BytestreamPath(input_path)

  @abstractmethod
  def get_files(self) -> t.Generator[t.Any]:
    """Returns a generator for the sequence of input files
    """
    pass

  def list_sources(self):
    return self.input_path.list_sources()

class TfrReader(BatchReader):

  """Class that reads a sequence of Tensorflow Record files as an instance of TFRecordDataset
  """

  def get_files(self):

    str_sources = [str(src) for src in self.input_path.list_sources()]

    return TFDatasetWrapper(filenames=str_sources)


class FileReader(BatchReader):

  """Class that reads a sequence of generic files 
  """

  def get_files(self):

    for src in self.input_path.list_sources():
      yield InMemoryFile(filename = str(src), content = src.read_bytes())

class ZipReader(BatchReader):

  """Class that reads a sequence of zip files 
  """

  def get_files(self):

    for src in self.input_path.list_sources():
      yield InMemoryZipHolder(filename = str(src), content = src.read_bytes())

class FontfileReader(BatchReader):

  """Class that reads a sequence of ttf/otf files 
  """

  def get_files(self):

    for src in self.input_path.list_sources():
      yield InMemoryFontfileHolder(filename = str(src), content = src.read_bytes())


class ScrapperReader(BatchReader):

  """Class that reads a sequence of bytes from scrapped urls

  Attributes:
      scrappers (t.List[Scrappers]): List of Scrapper instances
  """

  def __init__(self, scrappers: t.List[Scrapper]):

    self.scrappers = scrappers

  def get_files(self):

    for scrapper in self.scrappers:
      for url in scrapper.get_source_urls():
        stream = BytestreamPath(url)
        yield InMemoryFile(filename = str(stream), content = stream.read_bytes())



class ReaderClassFactory(object):

  """Factory class that returns the appropriate reader depending on the expected input file format.
  """
  
  def get(file_format: type) -> type:
    """Returns a reader type for instantiation at runtime.
    
    Args:
        file_format (InMemoryFile): Expected input file format.
    
    """
    if file_format == TFDatasetWrapper:
      return TfrReader
    elif file_format == InMemoryZipHolder:
      return ZipReader
    elif file_format == InMemoryFontfileHolder:
      return FontfileReader
    else:
      logger.info("Reader class defaulted to FileReader; reading uninterpreted bytestreams.")
      return FileReader