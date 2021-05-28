"""This module contains classes that deal with encoding/decoding bytestreams from/to the data formats that act as the interface between pipeline stages and storage, and between different ML pipeline stages.

"""
from __future__ import annotations
import zipfile
import typing as t
import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel

from numpy import ndarray
import imageio

from tensorflow import string as tf_str
from tensorflow.train import (Example as TFExample, Feature as TFFeature, Features as TFFeatures, BytesList as TFBytesList)
from tensorflow.io import FixedLenFeature, parse_single_example

logger = logging.getLogger(__name__)

class TfrWritable(ABC, BaseModel):

  """Class that provides Tensorflow record's encoding and decoding logic for downstream data formats used by the package
  """
  
  _tfr_schema: t.Dict

  @classmethod
  def bytes_feature(value: bytes) -> TFFeature:
    """Maps a bytestream to a TF Feature instance
    
    Args:
        value (bytes): bytes to encode
    
    Returns:
        TFFeature: encoded value
    """
    return TFFeature(bytes_list=TFBytesList(value=[value]))

  @classmethod
  @abstractmethod
  def to_tf_format(obj: TfrWritable) -> TFFeature:
    """Maps an object inheriting from this class to a TF record compatible format
    
    Args:
        obj (TfrWritable): Object to be encoded
    
    Returns:
        t.Dict: dictionary with encoded features that will be stored into a TF record.
    """


  def to_tfr(self):
    """Returns a Tensorflow example instance encoding the instance's contents
    
    """

    return TFExample(
      features = TFFeatures(features = cls.to_tf_format(self)))

  @classmethod
  def from_tfr(cls, serialised: TFExample) -> TfrWritable:
    """Creates a class' instance by deserialising a TF record using the class schema
    
    Args:
        serialised (TFExample): serialised TF example
    
    Returns:
        TfrWritable: deserialised TfrWritable instance
    """
    return parse_single_example(serialised,cls._tfr_schema)



class InMemoryFile(TfrWritable):

  """Wrapper class for retrieved file bytestreams
  """
  _tfr_schema: t.Dict = {
      "filename": FixedLenFeature([], tf_str)
      "content": FixedLenFeature([], tr_str)
  }

  filename: str
  content: bytes
  
  # def __init__(self, **data):

  #   filtered_data = data.pop("_tfr_schema")
  #   super().__init__(**filtered_data)

  def __eq__(self,other):
    return isinstance(other, InMemoryFile) and self.filename == other.filename and self.content == other.content 

  @classmethod
  def to_tf_format(obj: InMemoryFile) -> TFFeature:
    return {
    "filename": bytes_feature(bytes(str.encode(obj.filename))),
    "content": bytes_feature(obj.content)
    }





class LabeledExample(TfrWritable):
  # wrapper that holds a labeled ML example, with asociated metadata
  features: ndarray
  label: str
  fontname: str

  _tfr_schema: t.Dict = {
    'features': FixedLenFeature([], tf_str),
    'label': FixedLenFeature([], tf_str),
    'fontname': FixedLenFeature([], tf_str),
    }

  # def __init__(self, **data):

  #   filtered_data = data.pop("_tfr_schema")
  #   super().__init__(**filtered_data)


  def __iter__(self):
    return iter((self.features,self.label,self.fontname))

  def __eq__(self,other):
    return isinstance(other, LabeledExample) and self.features == other.features and self.label == other.label and self.fontname == other.fontname

  @classmethod
  def img_to_png_bytes(cls, img):
    bf = io.BytesIO()
    imageio.imwrite(bf,img,"png")
    val = bf.getvalue()
    bf.close()
    return val

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

  def to_tf_format(self) -> TFFeature:
    return {
    "label": bytes_feature(bytes(str.encode(self.label))),
    "fontname": bytes_feature(bytes(str.encode(self.fontname))),
    "features": bytes_feature(cls.img_to_png_bytes(self.features))
    }

  def to_in_memory_file(self):
    """Converts the instance to an InMemoryFile instance, saving the label and fontname in the file name and the features in the file content as bytes
    
    Returns:
        InMemoryFile: Reformatted object
    """
    # generate random tag to avoid possible collisions in filenames (useful for BatchWriters)
    random_tag = random.getrandbits(32)
    return InMemoryFile(filename = f"fontfile:{self.fontname},label:{self.label},feature_dims:{self.features.shape},random_tag:{random_tag}", content = self.features.tobytes())

  @classmethod
  def from_in_memory_file(cls, file: InMemoryFile):
    """Attempts to convert an instance of InMemoryFile to a LabeledExample object
    
    Args:
        file (InMemoryFile): file to convert
    
    Returns:
        LabeledExample: Reformatted object
    """
    
    rgx = re.compile(r"fontfile:(\w+),label:(.),feature_dims:\((\d+), (\d+)\)".+)
    try:
      fileontname, label, d1, d2 = re.findall(rgx, file.filename)[0]
      return LabeledExample(
      features = np.frombuffer(file.content).reshape((int(d1), int(d2))),
      label = label,
      fontname = fontname)
    except Exception as e:
      logging.exception(f"Error when trying to convert InMemoryFile to LabeledExample: {e}")





class InMemoryZipHolder(InMemoryFile):

  """In-memory buffer class for ingested zip bytestream
  """
  
  def to_zipfile(self):
    """
    
    Returns:
        zipfile.ZipFile: An open ZipFile instance with the current instance's contents
    """
    return zipfile.ZipFile(io.BytesIO(self.content),"r")



class InMemoryFontfileHolder(InMemoryFile):

  """In-memory buffer class for ingested ttf or otf font files
  """
  
  def to_truetype(self, font_size):
    """
    
    Returns:
        font: A parsed font object.
    """
    ImageFont.truetype(io.BytesIO(self.content),self.font_extraction_size)



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

  def add_file(self,file: InMemoryFile):
    """Add a file to the open zip file
    
    Args:
        file (InMemoryFile): file to be added
    """
    file_size = sys.getsizeof(file.content)
    self.zip_file.writestr(str(self.n_files) + file.filename, file.content)
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