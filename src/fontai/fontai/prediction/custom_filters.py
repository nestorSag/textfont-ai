"""
This module contains filtering functions for Tensorflow dataset operations that are applied to model inputs right after being deserialised to be used at training time
"""
import tensorflow as tf
import typing as t

__all__ = [
  "filter_misclassified_chars",
  "filter_chars_by_score",
  "filter_fonts_by_size",
  "filter_irregular_fonts",
  "filter_by_name"
]

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

def filter_fonts_by_size(n_chars: int):
  """Returns a Filtering function for Tensorflow datasets that filter out fonts with too few remaining characters.

  Returns:
      t.Callable: Filtering function for Tensorflow datasets
  """

  if n_chars < 0:
    raise ValueError("n_chars must be non-negative")

  def f(kwargs):
    """
    
    Args:
        kwargs (t.Dict): a dictionary with every object parsed from a serialised Tensorflow example, including "features" and "label" entries.
    
    Returns:
        boolean
    """
    return tf.size(kwargs["label"]) >= n_chars
    #return tf.logical_and(tf.size(kwargs["features"]) > 0, kwargs["features"].shape[0] >= n)

  return f


def filter_irregular_fonts(min_score: int):
  """Returns a Filtering function for Tensorflow datasets that filter out fonts with misclassified characters or low-confidence scores.
  
  
  Args:
      min_score (int): minimum score threshold

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
    predicted_label_idx = tf.argmax(kwargs["score"], axis=-1)
    predicted_labels = tf.gather(kwargs["charset_tensor"], predicted_label_idx, axis=-1)

    predicted_scores = tf.reduce_max(kwargs["score"], axis=-1)

    return tf.math.logical_and(tf.math.reduce_all(predicted_scores >= min_score), tf.math.reduce_all(predicted_labels == kwargs["label"]))
    

  return f


def filter_by_name(substring: str):
  """Returns a Filtering function for Tensorflow datasets that filter out fonts whose name does not contain the provided substring, e.g. italic, 3d, etc.
  
  
  Args:
      substring (str): substring that will be searched for in the lowercased font names

  Returns:
      t.Callable: Filtering function for Tensorflow datasets
  
  """
  lower_substring = substring.lower()

  def f(kwargs):
    """
    
    Args:
        kwargs (t.Dict): a dictionary with every object parsed from a serialised Tensorflow example, including "features" and "label" entries.
    
    Returns:
        boolean
    """
    return tf.strings.regex_full_match(tf.strings.lower(kwargs["fontname"]), f".*{lower_substring}.*")
    

  return f