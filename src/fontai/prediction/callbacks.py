"""
This module contains custom Tensorflow callbacks
"""
import os
import typing as t

import tensorflow as tf
from tensorflow.python.keras import backend
import matplotlib.pyplot as plt
import numpy as np
import mlflow

class SAAEImageSamplerCallback(tf.keras.callbacks.Callback):

  """Generates randomly chosen character images from the model at the end of each epoch and pushes them to MLFLow.
  
  """
  
  def __init__(
    self,
    n_labels: int,
    embedding_dim: int,
    n_imgs=16):
    """
    Args:
        n_labels (int): Number of labels in model's charset
        embedding_dim (int): Dimensionality of encoded representation
        n_imgs (int, optional): Number of images to sample
    
    
    """

    self.n_labels = n_labels
    self.embedding_dim = embedding_dim
    self.n_imgs = n_imgs

  def on_epoch_end(self,epoch,numpy_logs):
    output_file = f"{epoch}.png"
    imgs = self.generate_images()
    self.plot_images(imgs, output_file)
    mlflow.log_artifact(output_file, self.__class__.__name__)
    os.remove(output_file)

  def generate_images(self):

    # sample encoded representation
    samples = self.model.prior_sampler(shape=(self.n_imgs,self.embedding_dim)).numpy()
    # sample one hot encoded labels
    labels = []
    for k in range(self.n_imgs):
      label = np.random.randint(0,self.n_labels,1)
      onehot = np.zeros((1,self.n_labels), dtype=np.float32)
      onehot[0,label] = 1.0
      labels.append(np.concatenate([samples[k].reshape((1,self.embedding_dim)),onehot],axis=-1))
    
    fully_encoded = np.array(labels, dtype=np.float32)

    imgs = self.model.decoder.predict(fully_encoded)
    return imgs


  def plot_images(self, imgs: t.Union[tf.Tensor, np.ndarray], output_file: str, n_cols = 7) -> None:
    """Utility function to plot a sequence of characters and save it in a given location as a single tiled figure.
    
    Args:
        imgs (t.Union[tf.Tensor, np.ndarray]): 4-dimensional array of images
        output_file (str): output file
        n_cols (int, optional): number of columns in output figure
    """
    # plot multiple images
    n_imgs, height, width, c = imgs.shape
    n_rows = int(np.ceil(n_imgs/n_cols))

    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(n_imgs):
      x = imgs[i]
      if isinstance(x, np.ndarray):
        x_ = x
      else:
        x_ = x.numpy()
      np_x = (255 * x_).astype(np.uint8).reshape((height,width))
      axs[int(i/n_cols), i%n_cols].imshow(np_x)
    
    plt.savefig(output_file)

class SAAEFontSamplerCallback(SAAEImageSamplerCallback):

  """Generates a random font style from the model, generates all of its characters and pushes them to MLFLow at the end of each epoch
  
  """
  
  def __init__(
    self,
    n_labels: int,
    embedding_dim: int):
    """
    Args:
        n_labels (int): Number of labels in model's charset
        embedding_dim (int): Dimensionality of encoded representation
    
    """

    super().__init__(n_labels,embedding_dim,n_labels)

  def generate_images(self):

    # sample encoded representation
    sample = self.model.prior_sampler(shape=(1,self.embedding_dim)).numpy()
    # sample one hot encoded labels
    labels = []
    for k in range(self.n_labels):
      onehot = np.zeros((1,self.n_labels), dtype=np.float32)
      onehot[0,k] = 1.0
      labels.append(np.concatenate([sample,onehot],axis=-1))
    
    fully_encoded = np.array(labels, dtype=np.float32)

    imgs = self.model.decoder.predict(fully_encoded)
    return imgs

class SAAELRHalver(tf.keras.callbacks.Callback):

  """Halves the step size of every embedded model in a custom supervised adversarial autoencoder as defined in the `models` submodule, up to a minimum accepted step size
  
  Attributes:
      halve_after (int): number of epochs after which step sizes are halved
      min_lr (float): lower bound for step size
  """

  def __init__(self, halve_after: int = 10, min_lr: float = 0.0001):

    self.halve_after = 10
    self.min_lr = min_lr

  def on_epoch_begin(self, epoch, logs=None):

    for model in self.model.model_list:
      model_lr = getattr(self.model, model).optimizer.lr
      lr = float(backend.get_value(model_lr))
      lr = max(lr/2**int(epoch/self.halve_after), self.min_lr)
      backend.set_value(model_lr, backend.get_value(lr))
