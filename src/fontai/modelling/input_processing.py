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
from  tf.python.data.ops.dataset_ops import MapDataset

from fontai.core.base import TfrHandler

logger = logging.getLogger(__name__)

class LabeledExamplePreprocessor(object):
  """
    Fetches and processes a list of Tensorflow record files to be consumed by an ML model

    batch_size: training mini batches' size

    charset: One of 'lowercase', 'uppercase', 'digits' or 'all'; otherwise, chracters in the provided string are used as acustom charset.

    tfr_filters: Filtering functions for Tensorflow records before being casted as training examples

    example_filters: Filtering functions for sets of image tensors and one-hot-encoded labels
  """
  def __init__(
    self, 
    batch_size: int,
    charset: str,
    tfr_filters: t.List[TfrFilter] = [],
    filters: t.List[t.Callable] = []):

    self.CHARSET_OPTIONS = {
      "uppercase": string.ascii_letters[26::],
      "lowercase": string.ascii_letters[0,26],
      "digits": string.digits,
      "all": string.ascii_letters + string.digits
    }

    self.filters = filters

    try:
      self.charset = self.CHARSET_OPTIONS[charset]
    except KeyError as e:
      logger.warning(f"Charset string is not one from {list(self.CHARSET_OPTIONS.keys())}; creating custom charset from provided string instead.")
      self.charset = "".join(list(set(charset)))

    self.charset_encoding = tf.convert_to_tensor(list(self.charset))


  def fetch_tfr_files(self, dataset: TFRecordDataset, drop_fontname=True):
    """
      Fetches a list of input Tensorflow record files and prepares them for training

      input_files: List of input files

      Returns a MapDataset object
    """

    dataset = dataset\
      .map(LabeledExample.from_tfr)\
      .filter(self.filter_charset)

    dataset = dataset.map(self.parse_tf_records)

    dataset = self.batcher(dataset)

    for example_filter in self.filters:
      dataset = dataset.map(example_filter)

    if drop_fontname:
      dataset = dataset.map(self.drop_fontname)

    return dataset

  def batcher(self, dataset):
    """
      Scrambles a data set randomly and makes it unbounded in order to process an arbitrary number of batches
    """
    dataset = dataset\
      .shuffle(buffer_size=2*self.batch_size)\
      .repeat()\
      .batch(self.batch_size)

    return dataset

  def parse_tf_records(self, record):
    """
      Parses a serialised Tensorflow record to retrieve image tensors, one-hot-encoded labels and metadata

      record: Serialised Tensorflow record

      Returns a triplet with the deserialised object.
    """
    img = tf.image.decode_png(record["features"])
    img = tf.cast(img,dtype=tf.float32)
    one_hot_label = tf.cast(tf.where(self.charset_encoding == record["label"]),dtype=tf.int32)
    label = tf.reshape(tf.one_hot(indices=one_hot_label,depth=self.num_classes),(self.num_classes,))#.reshape((num_classes,))
    return img, label, record["fontname"]

  def filter_charset(self, img, label, metadata):
    """
      Filters out triplet examples not containing characters in the charset
    """
    return tf.reduce_any(self.tf_classes == parsed["label"])

  def drop_fontname(self, features, label, fontname):
    """
      Drops the metadata from the triplet sothe remaining tuple can be passed to a Tensorflow model.
    """
    return features, label





class InputFilterFactory(object):

  """Factory class for different filtering conditions that can be applied to the model's input before being used for training. Functions returned by the factory serve as input to InputPreprocessing instances
  
  """
  
  @classmethod
  def supervised_filter(self,threshold=0.5, model_path: str) -> t.Callable:
    """Returns a filtering function to filter out misclassfied examples or correctly classified examples that are not unambiguous enough. This is useful to filter out 'hard' examples for generative models
    
    Args:
        threshold (float, optional): Probability theshold below which correctly classified examples are also filtered out
        model_path (str): Path from which to load the scoring model
    
    Returns:
        t.Callable: Filtering function.
    """
    model = Model.load(model_path)

    def filter(features,labels,fontnames):
      # filters a batch using a trained model
      pred = model.predict(features)
      condition = tf.argmax(pred,axis=-1) == tf.argmax(labels,axis=-1) and tf.max(pred,axis=-1) > threshold
      return features[condition], labels[condition], fontnames[condition]

    return filter_func

  @classmethod
  def fontname_filter(self,regex: str) -> t.Callable:
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



