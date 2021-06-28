import tensorflow as tf

def filter_misclassified_in_font():
  """Returns a function for Tensorflow datasets that filter out misclassified examples in a font image set; examples must be a scored record.
  
  Returns:
      callable: Filtering function for Tensorflow datasets
  """
  def f(kwargs):
    index = tf.argmax(kwargs["label"], axis=-1) == tf.argmax(kwargs["score"], axis=-1)
    return {key: kwargs[key][index] for key in kwargs}

  return f

def filter_by_score_in_font(threshold: float):
  """Returns a function for Tensorflow datasets that filter out scores lower than a given threshold in a font image set
  
  Args:
      threshold (float): Score threshold
  """

  if threshold > 1 or threshold <= 0:
    raise ValueError("Threshold value must be in (0,1]")

  def f(kwargs):
    index = tf.reduce_max(kwargs["score"],axis=-1) >= threshold
    return {key: kwargs[key][index] for key in kwargs}

  return f
