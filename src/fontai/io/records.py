"""This module contains classes that deal with encoding/decoding bytestreams from/to the data formats that act as the interface between pipeline stages and storage, and between different ML pipeline stages.

"""
from __future__ import annotations
import zipfile
import typing as t
import io
import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel

from numpy import ndarray, uint8
import imageio

from tensorflow import string as tf_str, Tensor, executing_eagerly
from tensorflow.train import (Example as TFExample, Feature as TFFeature, Features as TFFeatures, BytesList as TFBytesList, FloatList as TFFloatList)
from tensorflow.io import FixedLenFeature, parse_single_example, serialize_tensor

logger = logging.getLogger(__name__)

class TfrWritable(ABC, BaseModel):

  """Class that provides Tensorflow record's encoding and decoding logic for downstream data formats used by the package
  """
  
  _tfr_schema: t.Dict

  def bytes_feature(self, value: bytes) -> TFFeature:
    """Maps a bytestream to a TF Feature instance
    
    Args:
        value (bytes): bytes to encode
    
    Returns:
        TFFeature: encoded value
    """
    return TFFeature(bytes_list=TFBytesList(value=[value]))

  def float_feature(self, value: float):
    """Maps a list of floats to a TF Feature instance
    
    Args:
        value (float): value
    
    Returns:
        TFFeature: encoded value
    """
    return tf.train.Feature(float_list=tf.train.TFFloatList(value=[value]))


  @classmethod
  @abstractmethod
  def serialise(self) -> TFFeature:
    """Maps an object inheriting from this class to a TF record compatible format
    
    Returns:
        t.Dict: dictionary with encoded features that will be stored into a TF record.
    """
    pass

  def to_tfr(self):
    """Returns a Tensorflow example instance encoding the instance's contents
    
    """

    return TFExample(
      features = TFFeatures(feature = self.serialise()))

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
    imageio.imwrite(bf,img.astype(uint8),"png")
    val = bf.getvalue()
    bf.close()
    return val

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

  def serialise(self) -> t.Dict:
    return {
    "label": self.bytes_feature(bytes(str.encode(self.label))),
    "fontname": self.bytes_feature(bytes(str.encode(self.fontname))),
    "features": self.bytes_feature(self.img_to_png_bytes(self.features))
    }


class ScoredExample(TfrWritable):
  # wrapper that holds a scored ML example and raw features
  features: ndarray
  score: ndarray 

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


  def serialise(self) -> t.Dict:
    # t = serialize_tensor(self.score)
    # if executing_eagerly():
    #   t_ = t.numpy()
    # else:
    #   t_ = t.eval()

    return {
    "score": self.bytes_feature(self.score.tobytes()),
    "features": self.bytes_feature(LabeledExample.img_to_png_bytes(self.features))
    }



class ScoredLabeledExample(TfrWritable):
  # wrapper that holds a scored, labeled ML example. Useful for model evaluation
  labeled_example: LabeledExample
  score: ndarray

  _tfr_schema: t.Dict = {**LabeledExample._tfr_schema, **{'score': FixedLenFeature([], tf_str)}}

  def __iter__(self):
    return iter((self.features,self.score))

  def __eq__(self,other):
    return isinstance(other, ScoredLabeledExample) and self.labeled_example == other.labeled_example and self.score == other.score

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

  def serialise(self) -> t.Dict:

    # t = serialize_tensor(self.score)
    # if executing_eagerly():
    #   t_ = t.numpy()
    # else:
    #   t_ = t.eval()

    return {
    **{"score": self.bytes_feature(self.score.tobytes())},
    **self.labeled_example.serialise()
    }

