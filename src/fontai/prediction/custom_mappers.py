"""
This module contains mapper functions to be applied to Tensorflow examples right after being deserialised to be used at training time; at the moment available functions filter examples based on the score's accuracy and values, and do so for scored font records.
"""

import tensorflow as tf
import typing as t

def drop_misclassified_in_font():
  """Returns a mapper function for Tensorflow datasets that drops misclassified images in a font batch; examples must have the schema as in ScoredLabeledChars._tfr_schema
  
  Returns:
      t.Callable: Mapping function for Tensorflow datasets
  """
  def f(kwargs: t.Dict):
    """

    Args:
        kwargs (t.Dict): a dictionary with every object parsed from a serialised Tensorflow example, including "features" and "label" entries.
    
    Returns:
        t.Dict: dictionary with filtered features and scores
    """
    #if label is empty, do nothings
    if tf.equal(tf.size(kwargs["label"]), 0):
      return kwargs

    predicted_label_idx = tf.math.reduce_max(kwargs["score"], axis=-1)
    predicted_labels = tf.gather(kwargs["charset_tensor"], predicted_label_idx, axis=-1)

    index = predicted_labels == kwargs["label"]

    if tf.equal(tf.reduce_sum(tf.cast(index, dtype=tf.int32)), 0):
      kwargs["label"] = tf.zeros((0,),dtype=tf.float32) #if no accurate scores are left, pass empty label for downstream deletion
      kwargs["score"] = kwargs["score"][index]
    else:
      kwargs["features"] = kwargs["features"][index]
      kwargs["label"] = kwargs["label"][index]
      kwargs["score"] = kwargs["score"][index]

    return kwargs

  return f

def keep_high_scores_in_font(threshold: float):
  """Returns a function for Tensorflow datasets that drops images with low classification score in a font batch
  
  Args:
      threshold (float): Score threshold

  Returns:
      t.Callable: Mapping function fot Tensorflow datasets
  """

  if threshold > 1 or threshold <= 0:
    raise ValueError("Threshold value must be in (0,1]")

  def f(kwargs: t.Dict):
    """
    
    Args:
        kwargs (t.Dict): a dictionary with every object parsed from a serialised Tensorflow example, including "features" and "label" entries.
    
    Returns:
        t.Dict: dictionary with filtered features and scores
    """
    #if label is empty, d nothings
    #if label is empty, do nothings
    if tf.equal(tf.size(kwargs["label"]), 0):
      return kwargs

    index = tf.reduce_max(kwargs["score"], axis=-1) >= score

    if tf.equal(tf.reduce_sum(tf.cast(index, dtype=tf.int32)), 0):
      kwargs["label"] = tf.zeros((0,),dtype=tf.float32) #if no accurate scores are left, pass empty label for downstream deletion
    else:
      kwargs["features"] = kwargs["features"][index]
      kwargs["label"] = kwargs["label"][index]
      kwargs["score"] = kwargs["score"][index]

    return kwargs

  return f
