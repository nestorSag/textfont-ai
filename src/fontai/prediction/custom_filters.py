"""
This module contains filtering functions for Tensorflow dataset operations that are applied to model inputs right after being deserialised to be used at training time
"""
import tensorflow as tf
import typing as t

__all__ = ["filter_misclassified_chars",
  "filter_chars_by_score",
  "filter_fonts_by_size"]

def filter_misclassified_chars():
  """Returns a filtering function for Tensorflow datasets that filter out misclassified examples; examples must have the schema as in ScoredLabeledChars._tfr_schema

  Returns:
      t.Callable: Filtering function for Tensorflow datasets
  """
  def f(kwargs):
    """
    
    Args:
        kwargs (t.Dict): a dictionary with every object parsed from a serialised Tensorflow example, including "features" and "label" entries.
    
    Returns:
        boolean
    """

    classification_index = tf.argmax(kwargs["score"], axis=-1)
    return kwargs["label"] == kwargs["charset_tensor"][classification_index]

  return f

def filter_chars_by_score(threshold: float):
  """Returns a Filtering function for Tensorflow datasets that filter out scores lower than a given threshold.

  Returns:
      t.Callable: Filtering function for Tensorflow datasets
  """

  if threshold > 1 or threshold <= 0:
    raise ValueError("Threshold value must be in (0,1]")

  def f(kwargs):
    """
    
    Args:
        kwargs (t.Dict): a dictionary with every object parsed from a serialised Tensorflow example, including "features" and "label" entries.
    
    Returns:
        boolean
    """
    return tf.reduce_max(kwargs["score"],axis=-1) >= threshold

  return f

def filter_fonts_by_size(n: int):
  """Returns a Filtering function for Tensorflow datasets that filter out fonts with too few remaining characters.

  Returns:
      t.Callable: Filtering function for Tensorflow datasets
  """

  if n < 0:
    raise ValueError("n must me non-negative")

  def f(kwargs):
    """
    
    Args:
        kwargs (t.Dict): a dictionary with every object parsed from a serialised Tensorflow example, including "features" and "label" entries.
    
    Returns:
        boolean
    """
    return tf.size(kwargs["label"]) >= n
    #return tf.logical_and(tf.size(kwargs["features"]) > 0, kwargs["features"].shape[0] >= n)

  return f

