import os
from pathlib import Path
import string
from fontai.pipeline.stages import Predictor

from tensorflow.data import TFRecordDataset

import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #this is needed to run models on GPU
charset = string.ascii_letters[26::]

def plot_img(x):
  if isinstance(x, np.ndarray):
    x_ = x
  else:
    x_ = x.numpy()
  np_x = (255 * x_).astype(np.uint8).reshape((64,64))
  plt.imshow(np_x)
  plt.show()

def generate(char = "A", embedded=None, scale=1):
  if embedded is None:
    embedded = np.random.normal(size=(1,10), scale=scale)
  #
  label = np.zeros((1,26), dtype=np.float32)
  index = charset.index(char)
  label[0,index] = 1.0
  return np.concatenate([embedded,label],axis=-1)

def canonical(k):
  x = np.zeros((1,10),dtype=np.float32)
  x[0,k] = 1.0
  return x
# def compare_reconstruction(k):
#   x, y = features[k], labels[k]
#   embedding = pred.model.encoder.predict(x)
#   extended_embedding = 

pred = Predictor.from_config_file("config/parameters/score-generative-uppercase.yaml")

data = [[str(x) for x in list(Path("data/uppercase-scored").iterdir())][0]]
tf_data = TFRecordDataset(data)

fetcher = pred.input_preprocessor(
      batch_size = pred.training_config.batch_size,
      charset = pred.charset,
      filters = pred.training_config.filters)

parsed = fetcher.fetch(tf_data)

batch = next(iter(parsed))
features, labels, scores = batch
embeddings = pred.model.encoder.predict(features)
extended_embedding = np.concatenate([embeddings, labels], axis=-1)
reconstructed = pred.model.decoder.predict(extended_embedding)


plot_img(features[4])
plot_img(reconstructed[0])


gen = generate()

r = np.random.normal(size=(1,10)); plot_img(pred.model.decoder.predict(generate(embedded=2*r)))