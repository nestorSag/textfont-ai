import tensorflow as tf

def filter_misclassified():
  """Returns a filtering function for Tensorflow datasets that filter out misclassified examples; examples must have the schema as in ScoredLabeledExamples._tfr_schema
  
  Returns:
      callable: Filtering function for Tensorflow datasets
  """
  def f(features, labels, scores):
    return tf.argmax(labels, axis=-1) == tf.argmax(scores, axis=-1)

  return f

def filter_score(threshold: float):
  """Returns a Filtering function for Tensorflow datasets that filter out scores lower than a given threshold.
  
  Args:
      threshold (float): Score threshold
  """

  if threshold > 1 or threshold <= 0:
    raise ValueError("Threshold value must be in (0,1]")

  def f(features, labels, scores):
    return tf.reduce_max(scores,axis=-1) >= threshold

  return f
