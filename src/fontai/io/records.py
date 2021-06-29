"""This module contains classes that can be serialised/deserialised to/from Tensorflow record files; they are used by the prediction stage for both training and scoring

"""
from __future__ import annotations
import zipfile
import typing as t
from typing import TypeVar, SupportsAbs, Generic
import io
import logging
from abc import ABC, abstractmethod

from collections import OrderedDict

from pydantic import BaseModel

from numpy import ndarray, uint8, all as np_all
import imageio

from tensorflow import string as tf_str, Tensor, executing_eagerly, convert_to_tensor
from tensorflow.train import (Example as TFExample, Feature as TFFeature, Features as TFFeatures, BytesList as TFBytesList, FloatList as TFFloatList)
from tensorflow.io import FixedLenFeature, parse_single_example, serialize_tensor

import tensorflow as tf


logger = logging.getLogger(__name__)

class TfrWritable(ABC):

  """Class that provides Tensorflow record's encoding and decoding logic for downstream data formats used by the package
  """
  
  _tfr_schema: t.Dict

  @classmethod
  def tensor_to_numpy(cls, x: Tensor) -> ndarray:
    """Converts Tensor to numpy array
    
    Args:
        x (Tensor): Input tensor
    
    Returns:
        ndarray: numpy array
    """

    if executing_eagerly():
      return x.numpy()
    else:
      return x.eval()

  @classmethod
  def array_to_bytes(cls, x: t.Union[Tensor, ndarray], dtype: type) -> bytes:
    """Converts an array, either from numpy or Tensorflow, to a stream of bytes to be serialized
    
    Args:
        x (t.Union[Tensor, ndarray]): Input array
        dtype: type of returned tensor
    
    Returns:
        bytes: serialized array
    """

    serialised_tensor = serialize_tensor(convert_to_tensor(x, dtype=dtype))

    byte_content = cls.tensor_to_numpy(serialised_tensor)

    return byte_content

  @classmethod
  def bytes_feature(cls, value: bytes) -> TFFeature:
    """Maps a bytestream to a TF Feature instance
    
    Args:
        value (bytes): bytes to encode
    
    Returns:
        TFFeature: encoded value
    """
    return TFFeature(bytes_list=TFBytesList(value=[value]))


  @classmethod
  @abstractmethod
  def to_bytes_dict(self) -> TFFeature:
    """Maps an object inheriting from this class to a TF record compatible format
    
    Returns:
        t.Dict: dictionary with encoded features that will be stored into a TF record.
    """
    pass

  def to_tf_example(self):
    """Returns a Tensorflow example instance encoding the instance's contents
    
    """

    return TFExample(
      features = TFFeatures(feature = self.to_bytes_dict()))

  @classmethod
  def from_tf_example(cls, example: Tensor) -> t.Dict:
    """Creates an instance by deserialising a TF record using the class schema
    
    Args:
        example (TFExample): example TF example
    
    Returns:
        TfrWritable: deserialised TfrWritable instance
    """
    return parse_single_example(example,cls._tfr_schema)

  @classmethod
  def img_to_png_bytes(cls, img):
    bf = io.BytesIO()
    imageio.imwrite(bf,img.astype(uint8),"png")
    val = bf.getvalue()
    bf.close()
    return val

  def add_score(self, score: Tensor) -> TfrWritable:
    """Adds a model's score and return the appropriate record instance
    
    Args:
        score (Tensor): Model record
    
    Returns:
        TfrWritable: scored record instance
    """
    return NotImplementError("Adding a score is not implemented for this schema.")

  @classmethod
  @abstractmethod
  def parse_bytes_dict(self, record):
    """Performs basic parsing of deserialised features and returns dict with the same keys as the tfr schema's ordered dict
    
    Args:
        record (tf.train.TFExample): Input record
    
    Returns:
        t.Dict: Output dictionary
    """
    pass


  @classmethod
  @abstractmethod
  def get_training_parser(
    cls, 
    charset_tensor: Tensor) -> t.Callable:
    """Returns a function that maps partially parsed objects as outputted by parse_bytes_dict to a (features, label) tuple for training consumption
    
    Args:
        charset_tensor (Tensor): tensor fo valid characters
    
    Returns:
        t.Callable: Parser function
    """
    pass

  @classmethod
  def from_parsed_bytes_dict(cls, kwargs: t.Dict):
    """Instantiate from a parsed bytes dict extracted from a Tensorflow record file
    
    Args:
        kwargs (t.Dict): Parsed dictionary
    
    Returns:
        TfrWritable
    """
    return cls(**{key: kwargs[key].numpy() for key in kwargs})





class ModelWithAnyType(BaseModel):

    # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True


class LabeledChar(TfrWritable, ModelWithAnyType):
  # wrapper that holds a labeled ML example, with asociated metadata
  features: ndarray
  label: str
  fontname: str


  _tfr_schema = OrderedDict([
    ('features', FixedLenFeature([], tf_str)),
    ('label', FixedLenFeature([], tf_str)),
    ('fontname', FixedLenFeature([], tf_str))])

  # def __init__(self, **data):

  #   filtered_data = data.pop("_tfr_schema")
  #   super().__init__(**filtered_data)


  def __iter__(self):
    return iter((self.features,self.label,self.fontname))

  def __eq__(self,other):
    return isinstance(other, LabeledChar) and np_all(self.features == other.features) and self.label == other.label and self.fontname == other.fontname

  def to_bytes_dict(self) -> t.Dict:
    return {
    "label": self.bytes_feature(bytes(str.encode(self.label))),
    "fontname": self.bytes_feature(bytes(str.encode(self.fontname))),
    "features": self.bytes_feature(self.img_to_png_bytes(self.features))
    }

  def add_score(self, score: ndarray) -> TfrWritable:

    return ScoredLabeledChar(example = self, score = score)


  @classmethod
  def parse_bytes_dict(cls, record):

    img = tf.image.decode_png(record["features"])
    img = tf.cast(img,dtype=tf.float32)/255.0 #rescaled image data

    record["features"] = img
    return record


  @classmethod
  def get_training_parser(
    cls, 
    charset_tensor: Tensor) -> t.Callable:

    def parser(kwargs):

      num_classes = len(charset_tensor)

      one_hot_label = tf.cast(tf.where(charset_tensor == kwargs["label"]),dtype=tf.int32)
      if tf.equal(tf.size(one_hot_label),0):
        label = tf.cast(one_hot_label, dtype=tf.float32) #if label not in current charset, pass empty label for downstream deletion
      else:
        label = tf.reshape(tf.one_hot(indices=one_hot_label,depth=num_classes),(num_classes,))
      
      return kwargs["features"], label

    return parser
  



class LabeledFont(TfrWritable, ModelWithAnyType):
  # wrapper that holds an entire font's character set
  features: ndarray
  label: ndarray
  fontname: str

  _tfr_schema = OrderedDict([
    ('features', FixedLenFeature([], tf_str)),
    ('label', FixedLenFeature([], tf_str)),
    ('fontname', FixedLenFeature([], tf_str))])

  # def __init__(self, **data):

  #   filtered_data = data.pop("_tfr_schema")
  #   super().__init__(**filtered_data)


  def __iter__(self):
    n = len(self.label)
    return (LabeledChar(
      features = self.features[k], 
      label = self.label[k], 
      fontname=self.fontname) for k in range(n))

  def __eq__(self,other):
    return isinstance(other, LabeledFont) and np_all(self.features == other.features) and np_all(self.label == other.label) and self.fontname == other.fontname


  def to_bytes_dict(self) -> t.Dict:

    feature_shape = self.features.shape

    # add channel dimension to feature
    return {
    "features": self.bytes_feature(self.array_to_bytes(self.features.reshape(feature_shape + (1,)), dtype=tf.uint8)),
    "label": self.bytes_feature(self.array_to_bytes(self.label, dtype=tf.string)),
    "fontname": self.bytes_feature(bytes(str.encode(self.fontname))),
    }

  def add_score(self, score: ndarray) -> TfrWritable:

    return ScoredLabeledFont(example = self, score = score)

  @classmethod
  def parse_bytes_dict(cls, record):
    imgs = tf.io.parse_tensor(record["features"], out_type=tf.uint8)
    imgs = tf.cast(imgs,dtype=tf.float32)/255.0 #rescaled image data
    label = tf.io.parse_tensor(record["label"], out_type=tf.string)

    record["features"] = imgs
    record["label"] = label
    return record

  @classmethod
  def get_training_parser(
    cls, 
    charset_tensor: Tensor) -> t.Callable:

    def parser(kwargs):
      num_classes = len(charset_tensor)

      raw_one_hot = tf.cast(
        tf.reshape(kwargs["label"], (-1,1)) == charset_tensor,
        dtype=tf.int32
      ) #one hot encoding with up to 62 columns

      index = tf.reduce_sum(raw_one_hot, axis=-1) > 0 # detect rows where all columns are zero (labels not in current charset)

      if tf.equal(tf.reduce_sum(tf.cast(index, dtype=tf.int32)), 0):
        features = kwargs["features"]
        label = tf.zeros((0,),dtype=tf.float32) #if no labels are in current charset, pass empty label for downstream deletion
      else:
        one_hot_label = tf.argmax(raw_one_hot[index]) # filter chars not in charset
        label = tf.reshape(tf.one_hot(indices=one_hot_label,depth=num_classes),(num_classes,-1)) #create restricted one hot encoding
        features = kwargs["features"][index]


      return features, label


    return parser

class ScoredRecordFactory(object):

  """Creates classes for scored TfrWritable records
  """
  
  @classmethod
  def create(cls, T: type):
    """Create a scored record's class
    
    Args:
        T (type): Subclass of TfrWritable
    
    Returns:
        TfrWritable: scored record class
    
    Raises:
        TypeError
    """
    if not issubclass(T, TfrWritable):
      raise TypeError("T must be a subclass of TfrWritable")
    else:
      class ScoredRecord(TfrWritable):
        #

        record_type = T
        _tfr_schema = {**record_type._tfr_schema, **{'score': FixedLenFeature([], tf_str)}}

        def __init__(self, example: TfrWritable, score: ndarray):
          if not isinstance(example, T) or not isinstance(score, ndarray):
            return TypeError(f"example must be an instance of {T} and score must be a numpy array")

          self.example = example
          self.score = score


        def __eq__(self,other):
          return self.example == other.example and np_all(self.score == other.score)
        #
        def to_bytes_dict(self) -> t.Dict:
          #
          return {
          **{"score": self.record_type.bytes_feature(self.record_type.array_to_bytes(self.score, dtype=tf.float32))},
          **self.example.to_bytes_dict()
          }
        #
        @classmethod
        def parse_bytes_dict(cls, record):
          parsed_record_bytes_dict = cls.record_type.parse_bytes_dict(record)
          score = tf.io.parse_tensor(record["score"], out_type=tf.float32)
          parsed_record_bytes_dict["score"] = score
          return parsed_record_bytes_dict 

        @classmethod
        def from_parsed_bytes_dict(cls, kwargs: t.Dict):

          kwargs = {key: kwargs[key].numpy() for key in kwargs}
          score = kwargs.pop("score")

          return cls(example = cls.record_type(**kwargs), score = score)

        @classmethod
        def get_training_parser(
          cls, 
          charset_tensor: Tensor) -> t.Callable:

          base_parser = cls.record_type.get_training_parser(charset_tensor=charset_tensor)

          def parser(kwargs):
            score = kwargs.pop("score")

            return base_parser(kwargs)

          return parser
        
      return ScoredRecord


ScoredLabeledChar = ScoredRecordFactory.create(LabeledChar)

ScoredLabeledFont = ScoredRecordFactory.create(LabeledFont)
