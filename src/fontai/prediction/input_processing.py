"""Summary

Attributes:
    logger (TYPE): Description
"""
from __future__ import absolute_import
from collections.abc import Iterable
import os
import logging
import string
import zipfile
import io
import typing as t
import types
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import imageio
import tensorflow as tf
from  tensorflow.python.data.ops.dataset_ops import MapDataset

from tensorflow.data import TFRecordDataset

from fontai.io.records import LabeledExample

logger = logging.getLogger(__name__)

class LabeledExamplePreprocessor(object):
  """
  Fetches and processes a list of Tensorflow record files to be consumed by an ML model
  
  Attributes:
      batch_size (int): batch size
      charset (char): string with every character that needs to be extracted
      CHARSET_OPTIONS (TYPE): Dictionary from allowed charsets names to charsets
      charset_tensor (tf.Tensor): charset in tensor form
      filters (t.List[types.Callable]): List of filters to apply to training data
      num_classes (int): number of classes in charset
  """
  def __init__(
    self, 
    batch_size: int,
    charset: str,
    filters: t.List[t.Callable] = []):
    """
    
    Args:
        batch_size (int): training batch size
        charset (str): One of 'lowercase', 'uppercase', 'digits' or 'all'; otherwise, chracters in the provided string are used as acustom charset.

        filters (t.List[t.Callable], optional): Filtering functions for sets of image tensors and one-hot-encoded labels
    """
    self.batch_size = batch_size

    self.CHARSET_OPTIONS = {
      "uppercase": string.ascii_letters[26::],
      "lowercase": string.ascii_letters[0:26],
      "digits": string.digits,
      "all": string.ascii_letters + string.digits
    }

    self.filters = filters

    try:
      self.charset = self.CHARSET_OPTIONS[charset]
    except KeyError as e:
      logger.warning(f"Charset string is not one from {list(self.CHARSET_OPTIONS.keys())}; creating custom charset from provided string instead.")
      self.charset = "".join(list(set(charset)))

    self.num_classes = len(self.charset)
    self.charset_tensor = tf.convert_to_tensor(list(self.charset))


  def fetch(self, dataset: TFRecordDataset, training_format=True):
    """
    Fetches a list of input Tensorflow record files and prepares them for training or scoring
    
    dataset: List of input files
    
    training_format: if True, returns features and one hot encoded labels; otherwise, returns features, char labels and fontname
    
    Returns a MapDataset object
    
    Args:
        dataset (TFRecordDataset): input data
        training_format (bool, optional): If True, returns features and a one hot encoded label; otherwise, returns features, label (as a single character), and fontname
    
    Returns:
        TFRecordDataset: Dataset ready for model consumption
    """
    if training_format:
      formatter = self.parse_for_training
    else:
      formatter = self.parse_for_scoring

    dataset = dataset\
      .map(LabeledExample.from_tfr)\
      .filter(self.filter_charset)\
      .map(formatter)

    if training_format:
      for example_filter in self.filters:
        dataset = dataset.map(example_filter)

      dataset = dataset.map(self.drop_fontname)

    return self.batch_dataset(dataset, repeat=training_format)


  def batch_dataset(self, dataset, buffered_batches = 256, repeat=True):
    """
    Scrambles a data set randomly and makes it unbounded in order to process an arbitrary number of batches
    
    Args:
        dataset (TFRecordDataset): Input dataset
        buffered_batches (int, optional): Number of batches to fetch in memory buffer
        repeat (boolean): If true, make it a cyclical dataset
    
    Returns:
        TFRecordDataset
    """

    dataset = dataset\
      .shuffle(buffer_size=buffered_batches*self.batch_size)

    if repeat:
      dataset = dataset.repeat()

    return dataset.batch(self.batch_size)

  def parse_for_training(self, record):
    """
    Parses a serialised Tensorflow record to retrieve image tensors, one-hot-encoded labels and fontname
    
    Args:
        record (tf.train.TFExample): Input example
    
    Returns:
        t.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: output triplet
    """
    img = tf.image.decode_png(record["features"])
    img = tf.cast(img,dtype=tf.float32)
    one_hot_label = tf.cast(tf.where(self.charset_tensor == record["label"]),dtype=tf.int32)
    label = tf.reshape(tf.one_hot(indices=one_hot_label,depth=self.num_classes),(self.num_classes,))

    return img, label, record["fontname"]
  
  def parse_for_scoring(self, record):
    """
    Parses a serialised Tensorflow record to retrieve image tensors, char labels and fontname
    
    Args:
        record (tf.train.TFExample): Input example
    
    Returns:
        t.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: output triplet
    """
  
    img = tf.image.decode_png(record["features"])
    img = tf.cast(img,dtype=tf.float32)
    label = record["label"]

    return img, label, record["fontname"]

  def filter_charset(self, record):
    """
    Filters out triplet examples not containing characters in the charset
    
    Args:
        record (tf.train.TFExample): Input example
    
    Returns:
        Tensor: boolean
    """
    return tf.reduce_any(self.charset_tensor == record["label"])

  def drop_fontname(self, features, label, fontname):
    """
    Drops the fontname from the triplet sothe remaining tuple can be passed to a Tensorflow model.
    
    Args:
        features (Tensor)
        label (Tensor)
        fontname (Tensor)
    
    Returns:
        t.Tuple[tf.Tensor, tf.Tensor]: filtered tuple
    """
    return features, label



# def supervised_filter(model_path: str, threshold=0.5) -> t.Callable:
#   """Returns a filtering function to filter out misclassfied examples or correctly classified examples that are not unambiguous enough. This is useful to filter out 'hard' examples for generative models
  
#   Args:
#       threshold (float, optional): Probability theshold below which correctly classified examples are also filtered out
#       model_path (str): Path from which to load the scoring model
  
#   Returns:
#       t.Callable: Filtering function.
#   """
#   model = Model.load(model_path)

#   def filter(features,labels,fontnames):
#     # filters a batch using a trained model
#     pred = model.predict(features)
#     condition = tf.argmax(pred,axis=-1) == tf.argmax(labels,axis=-1) and tf.max(pred,axis=-1) > threshold
#     return features[condition], labels[condition], fontnames[condition]

#   return filter_func

def fontname_filter(regex: str) -> t.Callable:
  """Filter out examples whose filename foesn't match the provided regex. This is useful to isolate examples of differnet font types, e.g. bold, italic and 3D.
  
  Args:
      regex (str): Regex to be matched to fontnames
  
  Returns:
      t.Callable: Filtering function
  """
  def filter(features, labels, fontnames):
    condition = tf.strings.regex_full_match(fotnames, regex)
    return features[condition], labels[condition], fontnames[condition]

  return filter_func



