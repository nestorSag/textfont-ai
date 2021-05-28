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

class InputPreprocessor(object):
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
    example_filters: t.List[ExampleFilter] = []):

    self.CHARSET_OPTIONS = {
      "uppercase": string.ascii_letters[26::],
      "lowercase": string.ascii_letters[0,26],
      "digits": string.digits,
      "all": string.ascii_letters + string.digits
    }

    self.tfr_handler = TfrHandler()
    self.tfr_filters = tfr_filters
    self.example_filters = example_filters

    try:
      self.charset = self.CHARSET_OPTIONS[charset]
    except KeyError as e:
      logger.warning(f"Charset string is not one from {list(self.CHARSET_OPTIONS.keys())}; creating custom charset from provided string instead.")
      self.charset = "".join(list(set(charset)))

    self.charset_encoding = tf.convert_to_tensor(list(self.charset))


  def fetch_tfr_files(self, dataset: TFRecordDataset):
    """
      Fetches a list of input Tensorflow record files and prepares them for training

      input_files: List of input files

      Returns a MapDataset object
    """

    dataset = dataset\
      .map(LabeledExample.from_tfr)\
      .filter(self.filter_charset)

    for tfr_filter in self.tfr_filters:
      dataset = dataset.map(tfr_filter.get_filter)

    dataset = dataset.map(self.parse_tf_records)

    for example_filter in self.example_filters:
      dataset = dataset.map(example_filter.get_filter)

    dataset = dataset.map(self.drop_metadata)
    dataset = self.scrambler(dataset)

    return dataset

  def scrambler(self, dataset):
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
    img = tf.image.decode_png(record["image"])
    img = tf.cast(img,dtype=tf.float32)
    one_hot_label = tf.cast(tf.where(self.charset_encoding == record["label"]),dtype=tf.int32)
    label = tf.reshape(tf.one_hot(indices=one_hot_label,depth=self.num_classes),(self.num_classes,))#.reshape((num_classes,))
    return img, label, record["metadata"]

  def filter_charset(self, img, label, metadata):
    """
      Filters out triplet examples not containing characters in the charset
    """
    return tf.reduce_any(self.tf_classes == parsed["char"])

  def drop_metadata(self, img, label, metadata):
    """
      Drops the metadata from the triplet sothe remaining tuple can be passed to a Tensorflow model.
    """
    return img, label


class Filter(object):

  def get_filter(self) -> t.Callable:
    pass

class SupervisedFilter(object):

  def __init__(self, model_path):
    self.model = Model.load(model_path)

  def get_filter(self):

    def filter(imgs,labels,metadata):
      # filters a batch using a trained model
      pred = self.model(imgs)
      condition = tf.argmax(pred,axis=-1) == tf.argmax(labels,axis=-1)
      return imgs[condition], labels[condition], filenames[condition]

    return filter_func
