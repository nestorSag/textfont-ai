"""
This module contains input preprocessing logic that happends right before data is ingested by the model to be trained.
"""
from __future__ import absolute_import
from collections.abc import Iterable
import os
import logging
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
      charset_tensor (tf.Tensor): charset in tensor form
      custom_filters (t.List[types.Callable]): List of custom_filters to apply to training data
      num_classes (int): number of classes in charset
  """

  def __init__(
    self, 
    input_record_class: type,
    batch_size: t.Optional[int],
    charset_tensor: tf.Tensor,
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

    self.charset_tensor = tf.convert_to_tensor(charset_tensor)


  def fetch(self, dataset: TFRecordDataset, training_format=True):
    """
    Fetches a list of input Tensorflow record files and prepares them for training or scoring
    
    dataset: List of input files
    
    training_format: if True, returns features and one hot encoded labels; otherwise, returns a dict of parsed bytestreams with labels as bytes
    
    Returns a MapDataset object
    
    Args:
        dataset (TFRecordDataset): input data
        training_format (bool, optional): If True, returns features and a one hot encoded label; otherwise, returns a dict of parsed bytestreams with labels as bytes
    
    Returns:
        TFRecordDataset: Dataset ready for model consumption
    """

    # bytes -> dict -> tuple of objs
    dataset = dataset\
      .map(self.input_record_class.from_tf_example)\
      .map(self.input_record_class.parse_bytes_dict)#\
      #.filter(self.filter_charset)

    # apply custom filters to formatted tuples
    for example_filter in self.custom_filters:
        dataset = dataset.filter(example_filter)

    # apply custom map to formatted tuples
    for example_mapper in self.custom_mappers:
        dataset = dataset.filter(example_mapper)
    
    # if for training, take only features and formatted labels, and batch together
    if training_format:
      dataset = dataset\
        .map(self.input_record_class.get_training_parser(charset_tensor = self.charset_tensor))\
        .filter(self.label_is_nonempty) #enmpty labels signal something went wrong while parsing

      #dataset = dataset.map(self.trim_for_training)\
        #.filter(self.label_is_nonempty) #enmpty labels signal something went wrong while parsing

      dataset = self.batch_dataset(dataset)
      dataset = self.add_batch_shape_signature(dataset)
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

  def add_batch_shape_signature(self, data: TFRecordDataset) -> TFRecordDataset:
    """Intermediate method required to make training data shapes known at graph compile time. Returns the passed data wrapped in a callable object with explicit output shape signatures
    
    Args:
        data (TFRecordDataset): Input training data
    
    Returns:
        TFRecordDataset
    
    Raises:
        ValueError
    """
    def callable_data():
      return data

    features, labels = next(iter(data))
    # drop batch size form shape tuples
    ftr_shape = features.shape[1::]
    lbl_shape = labels.shape[1::]

    if len(ftr_shape) != 3 or len(lbl_shape) != 1:
      raise ValueError(f"Input shapes don't match expected: got shapes {features.shape} and {labels.shape}")

    training_data = tf.data.Dataset.from_generator(
      callable_data, 
      output_types = (
        features.dtype, 
        labels.dtype
      ),
      output_shapes=(
        tf.TensorShape((None,) + ftr_shape),
        tf.TensorShape((None,) + lbl_shape)
      )
    )

    return training_data