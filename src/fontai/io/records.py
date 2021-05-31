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


class ScoredExample(TfrWritable):
  # wrapper that holds a scored ML example
  features: ndarray
  score: np.float32

  _tfr_schema: t.Dict = {
    'features': FixedLenFeature([], tf_str),
    'score': FixedLenFeature([], tf_str)
  }

  # def __init__(self, **data):

  #   filtered_data = data.pop("_tfr_schema")
  #   super().__init__(**filtered_data)


  def __iter__(self):
    return iter((self.features,self.score))

  def __eq__(self,other):
    return isinstance(other, ScoredExample) and self.features == other.features and self.score == other.score

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

  def to_tf_format(self) -> TFFeature:
    return {
    "score": bytes_feature(bytes(self.score)),
    "features": bytes_feature(cls.img_to_png_bytes(self.features))
    }

