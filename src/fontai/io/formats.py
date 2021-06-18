"""This module contains classes that deal with encoding/decoding bytestreams from/to the data formats that act as the interface between pipeline stages and storage, and between different ML pipeline stages.

"""
from __future__ import annotations
import zipfile
import sys
import typing as t
import logging
import io
from abc import ABC, abstractmethod

from pydantic import BaseModel

from numpy import ndarray
import imageio
from PIL import ImageFont


#from tensorflow import string as tf_str
#from tensorflow.train import (Example as TFExample, Feature as TFFeature, Features as TFFeatures, BytesList as TFBytesList)
from tensorflow.data import TFRecordDataset

from tensorflow.io import FixedLenFeature, parse_single_example

logger = logging.getLogger(__name__)


class InMemoryFile(BaseModel):

  """Wrapper class for retrieved file bytestreams
  """

  filename: str
  content: bytes


  def to_format(self, file_format: InMemoryFile) -> InMemoryFile:
    """Cast instance as a different file type inehriting from InMemoryFile 
    
    Args:
        file_format (InMemoryFile): Target file type
    
    Returns:
        InMemoryFile: cast file instance.
    """
    return file_format(**self.dict())

  def deserialise(self):
    """Parse instance as the corresponding Python object such as a ZipFile or an ImageFont.truetype
    
    """
    return self

  @classmethod
  def serialise(cls, obj: t.Any):
    """Serialises a Python object to an instance of InMemoryFile
    
    Args:
        obj (t.Any): Python object
    
    Returns:
        InMemoryFile: Description
    
    Raises:
        TypeError: If passed object is not an instance of InMemoryFile
    """
    if not isinstance(obj, InMemoryFile):
      raise TypeError("InMemoryFile only can be serialised from an instance of the same class.")
    else:
      return obj

  def __str__(self):
    return f"Filename: {self.filename}, content size: {sys.getsizeof(self.content)/1e6} MB."



class InMemoryZipHolder(InMemoryFile):

  """In-memory buffer class for ingested zip bytestream
  """
  
  def deserialise(self):
    """
    
    Returns:
        zipfile.ZipFile: An open ZipFile instance with the current instance's contents
    """
    return zipfile.ZipFile(io.BytesIO(self.content),"r")

  @classmethod
  def serialise(self, obj: zipfile.ZipFile):
    raise NotImplementError("Serialisation to InMemoryZipHolder is not implemented.")
    # bf = io.BytesIO()
    # zipped = zipfile.ZipFile(bf,"w")
    # for file in obj.filelist():
    #   zipped.writestr(file, obj.read(file))

    # return InMemoryZipHolder(filename="holder", content = bf.getvalue())



class InMemoryFontfileHolder(InMemoryFile):

  """In-memory buffer class for ingested ttf or otf font files
  """
  
  def deserialise(self, font_size: int):
    """
    Args:
        font_size (int)
    
    No Longer Returned:
        font: A parsed font object.
    """
    return ImageFont.truetype(io.BytesIO(self.content),font_size)

  def serialise(self, font: ImageFont.FreeTypeFont):
    raise NotImplementError("Serialisation to InMemoryFontfileHolder is not implemented.")



# class TFDatasetWrapper(TFRecordDataset):

#   def to_format(self, file_format: type):
#     if file_format != TFDatasetWrapper:
#       raise TypeError("Tensorflow datasets cannot be converted to other custom file types.")

#     return self

#   def deserialise(self):
#     return self


class InMemoryZipBundler(object):

  """Class to fill a zipfile in memory before persisting it to storage.
  
  Attributes:
      buffer (BytesIO): zip file's buffer
      n_files (int): number of files currently in the zip file
      size (int): zip file's size
      zip_file (ZipFile): ZipFile instance wrapping the buffer
  """
  
  def __init__(self):
    self.size = 0
    self.n_files = 0

    self.buffer = io.BytesIO()
    self.zip_file = zipfile.ZipFile(self.buffer,"w")

  def write(self,file: InMemoryFile):
    """Add a file to the open zip file
    
    Args:
        file (InMemoryFile): file to be added
    """
    file_size = sys.getsizeof(file.content)
    self.zip_file.writestr(file.filename, file.content)
    self.n_files += 1
    self.size += file_size

  def compress(self):
    """Compress and close zip file
    
    Returns:
        InMemoryZipBundler: self
    """
    self.zip_file.close()
    return self

  def close(self):
    """Closes the zip file's inner buffer
    """
    self.buffer.close()

  def get_bytes(self):
    """Get zip file contents
    
    Returns:
        bytes: contents
    """
    return self.buffer.getvalue()