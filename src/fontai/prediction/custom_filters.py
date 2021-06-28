import tensorflow as tf

def filter_misclassified_chars():
  """Returns a filtering function for Tensorflow datasets that filter out misclassified examples; examples must have the schema as in ScoredLabeledChars._tfr_schema
  
  Returns:
      callable: Filtering function for Tensorflow datasets
  """
  def f(kwargs):
    return tf.argmax(kwargs["label"], axis=-1) == tf.argmax(kwargs["scores"], axis=-1)

  return f

def filter_by_score(threshold: float):
  """Returns a Filtering function for Tensorflow datasets that filter out scores lower than a given threshold.
  
  Args:
      threshold (float): Score threshold
  """

  if threshold > 1 or threshold <= 0:
    raise ValueError("Threshold value must be in (0,1]")

  def f(kwargs):
    return tf.reduce_max(kwargs["scores"],axis=-1) >= threshold

  return f
