"""
This module contains input preprocessing logic that happends right before data is ingested by the model to be trained.
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

import fontai.prediction.models as custom_models

logger = logging.getLogger(__name__)

class RecordPreprocessor(object):
  """
  Fetches and processes a list of Tensorflow record files to be consumed by an ML model
  
  Attributes:
      batch_size (int): batch size
      charset (char): string with every character that needs to be extracted
      CHARSET_OPTIONS (TYPE): Dictionary from allowed charsets names to charsets
      charset_tensor (tf.Tensor): charset in tensor form
      custom_filters (t.List[types.Callable]): List of custom_filters to apply to training data
      num_classes (int): number of classes in charset
  """

  CHARSET_OPTIONS = {
    "uppercase": string.ascii_letters[26::],
    "lowercase": string.ascii_letters[0:26],
    "digits": string.digits,
    "all": string.ascii_letters + string.digits
    }

  def __init__(
    self, 
    input_record_class: type,
    batch_size: t.Optional[int],
    charset: str,
    custom_filters: t.List[t.Callable] = [],
    custom_mappers: t.List[t.Callable] = []):
    """
    
    Args:
        batch_size (int): training batch size
        charset (str): One of 'lowercase', 'uppercase', 'digits' or 'all'; otherwise, chracters in the provided string are used as acustom charset.

        custom_filters (t.List[t.Callable], optional): Filtering functions for sets of image tensors and one-hot-encoded labels
    """

    self.input_record_class = input_record_class

    self.batch_size = batch_size

    self.custom_filters = custom_filters

    self.custom_mappers = custom_mappers

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
    
    training_format: if True, returns features and one hot encoded labels; otherwise, returns TfrWritable record instances
    
    Returns a MapDataset object
    
    Args:
        dataset (TFRecordDataset): input data
        training_format (bool, optional): If True, returns features and a one hot encoded label; otherwise, returns features, label (as a single character), and fontname
    
    Returns:
        TFRecordDataset: Dataset ready for model consumption
    """

    # bytes -> dict -> tuple of objs
    dataset = dataset\
      .map(self.input_record_class.from_tf_example)\
      .map(self.input_record_class.parse_bytes_dict)#\
      #.filter(self.filter_charset)
    
    #apply custom filters to raw tuples
    for example_filter in self.custom_filters:
        dataset = dataset.filter(example_filter)

    # apply custom map to raw tuples
    for example_mapper in self.custom_mappers:
        dataset = dataset.filter(example_mapper)

    # if for training, take only features and formatted labels, and batch together
    if training_format:
      dataset = dataset\
        .map(self.input_record_class.get_training_parser(charset_tensor = self.charset_tensor))\
        .filter(self.label_is_nonempty) #enmpty labels signal something went wrong while parsing

      dataset = self.batch_dataset(dataset)
    return dataset


  def batch_dataset(self, dataset, buffered_batches = 512):
    """
    Scrambles a data set randomly and makes it unbounded in order to process an arbitrary number of batches
    
    Args:
        dataset (TFRecordDataset): Input dataset
        buffered_batches (int, optional): Number of batches to fetch in memory buffer
    
    Returns:
        TFRecordDataset
    """

    buffer_size = buffered_batches*self.batch_size if self.batch_size is not None else 1000

    dataset = dataset\
      .shuffle(buffer_size=buffer_size)

    dataset = dataset.repeat()

    if self.batch_size is not None:
      return dataset.batch(self.batch_size)
    else:
      return dataset

  def label_is_nonempty(self, features, label):
    """
    Filters out training examples without rows or correctly formatted labels
    
    Args:
        record (tf.train.TFExample): Input example
    
    Returns:
        Tensor
    """
    return tf.math.logical_not(tf.equal(tf.size(label), 0))
