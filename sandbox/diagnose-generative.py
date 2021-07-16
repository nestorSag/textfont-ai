import os
from pathlib import Path
import string
from fontai.runners.stages import Predictor

from tensorflow.data import TFRecordDataset

import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #this is needed to run models on GPU
charset = string.ascii_letters[26::]

def plot_imgs(imgs):
  # plot multiple images
  fig, axs = plt.subplots(4,7)
  for i in range(26):
    x = imgs[i]
    if isinstance(x, np.ndarray):
      x_ = x
    else:
      x_ = x.numpy()
    np_x = (255 * x_).astype(np.uint8).reshape((64,64))
    #np_x[np_x <= np.quantile(np_x,0.25)] = 0.0
    axs[int(i/7), i%7].imshow(np_x)
  plt.show()

def plot_img(x):
  # plot single image
  if isinstance(x, np.ndarray):
    x_ = x
  else:
    x_ = x.numpy()
  np_x = (255 * x_).astype(np.uint8).reshape((64,64))
  plt.imshow(np_x)
  plt.show()

def generate(embedded=None):
  # generate a batch with all character in the alphabet
  if embedded is None:
    embedded = np.random.normal(size=(1,10))
  #
  single = []
  for index in range(26):
    label = np.zeros((1,26), dtype=np.float32)
    label[0,index] = 1.0
    single.append(np.concatenate([embedded,label],axis=-1))
  return np.concatenate(single,axis=0)

# def canonical(k):
#   x = np.zeros((1,10),dtype=np.float32)
#   x[0,k] = 1.0
#   return x
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
features, labels = batch
image_precode = pred.model.image_encoder.predict(features)
embeddings = pred.model.full_encoder.predict(np.concatenate([image_precode, labels.numpy()], axis=-1))
extended_embedding = np.concatenate([embeddings, labels], axis=-1)
reconstructed = pred.model.decoder.predict(extended_embedding)


plot_imgs(pred.model.decoder.predict(generate()))