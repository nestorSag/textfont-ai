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

from tensorflow.io import FixedLenFeature, parse_single_example
from fontai.io.storage import BytestreamPath

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
  def from_file(cls, filepath: str) -> InMemoryFile:
    """Instantiate from file path
    
    Args:
        filepath (str): FIle path
    
    Returns:
        InMemoryFile: instantiated object
    """
    return cls(filename = filepath, content = BytestreamPath(filepath).read_bytes())

  @classmethod
  def from_bytestream_path(cls, bsp: BytestreamPath) -> InMemoryFile:
    """Instantiate from bytestream path
    
    Returns:
        InMemoryFile: instantiated object
    
    """
    return cls(filename = str(bsp), content = bsp.read_bytes())

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



class InMemoryZipfile(InMemoryFile):

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
    raise NotImplementError("Serialisation to InMemoryZipfile is not implemented.")
    # bf = io.BytesIO()
    # zipped = zipfile.ZipFile(bf,"w")
    # for file in obj.filelist():
    #   zipped.writestr(file, obj.read(file))

    # return InMemoryZipfile(filename="holder", content = bf.getvalue())



class InMemoryFontfile(InMemoryFile):

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
    raise NotImplementError("Serialisation to InMemoryFontfile is not implemented.")