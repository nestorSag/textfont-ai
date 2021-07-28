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

    self.past_epochs = None

  def on_epoch_end(self,epoch,numpy_logs):

    if self.past_epochs is None:
      # determine whether there are previously generated images form previous runs
      active_run = mlflow.active_run()
      client = mlflow.tracking.MlflowClient()
      self.past_epochs = len(client.list_artifacts(active_run.info.run_id, self.__class__.__name__))

    logical_epoch = epoch + self.past_epochs
    output_file = f"{logical_epoch}.png"
    imgs = self.generate_images()
    self.plot_images(imgs, output_file)
    mlflow.log_artifact(output_file, self.__class__.__name__)
    os.remove(output_file)

  def generate_images(self):
    """Produces character images stacked in a tensor from a generative decoder model
    
    Returns:
        tf.Tensor
    """
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
    # images are along first axis (image <=> example)
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
      axs[int(i/n_cols), i%n_cols].imshow(np_x, cmap="Greys")

    # get rid of subfigure axes
    for k in range(n_rows*n_cols):
      axs[int(k/n_cols), k%n_cols].axis("off")
    
    plt.savefig(output_file)

class SAAEFontSamplerCallback(SAAEImageSamplerCallback):

  """Generates a random font style from the model, generates all of its characters and pushes them to MLFLow at the end of each epoch. This callback assumes the model generates a single font's character at a time, depending on the provided input label
  
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



class TensorSAAEFontSamplerCallback(SAAEImageSamplerCallback):

  """Generates a random font style from the model, generates all of its characters and pushes them to MLFLow at the end of each epoch. This callback assumes the model produces all character images as a 3d Tensor where each channel represent one of the font's characters.
  
  """
  
  def __init__(
    self,
    embedding_dim: int):
    """
    Args:
        n_labels (int): Number of labels in model's charset
        embedding_dim (int): Dimensionality of encoded representation
    
    """

    super().__init__(None,embedding_dim,None)

  def generate_images(self):
    """Generates images as Tensor. Images are along the last axis (images <=>channels)
    
    Returns:
        tf.Tensor
    """
    # sample encoded representation
    samples = self.model.prior_sampler(shape=(1,self.embedding_dim)).numpy()
    imgs = self.model.decoder.predict(samples)
    return imgs


  def plot_images(self, imgs: t.Union[tf.Tensor, np.ndarray], output_file: str, n_cols = 7) -> None:
    """Utility function to plot a sequence of characters and save it in a given location as a single tiled figure.
    
    Args:
        imgs (t.Union[tf.Tensor, np.ndarray]): 4-dimensional array of images
        output_file (str): output file
        n_cols (int, optional): number of columns in output figure
    """
    # plot multiple images
    n_fonts, height, width, n_imgs = imgs.shape
    n_rows = int(np.ceil(n_imgs/n_cols))

    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(n_imgs):
      x = imgs[0,:,:,i]
      if isinstance(x, np.ndarray):
        x_ = x
      else:
        x_ = x.numpy()
      np_x = (255 * x_).astype(np.uint8).reshape((height,width))
      axs[int(i/n_cols), i%n_cols].imshow(np_x, cmap="Greys")

    # get rid of subfigure axes
    for k in range(n_rows*n_cols):
      axs[int(k/n_cols), k%n_cols].axis("off")
    
    plt.savefig(output_file)




class SAAELRHalver(tf.keras.callbacks.Callback):

  """Halves the step size of every embedded model in a custom supervised adversarial autoencoder as defined in the `models` submodule, up to a minimum accepted step size
  
  Attributes:
      halve_after (int): number of epochs after which step sizes are halved
      min_lr (float): lower bound for step size
  """

  def __init__(self, halve_after: int = 10, min_lr: float = 0.0001):

    self.halve_after = 10
    self.min_lr = min_lr
    self.initial_lr = None

  def on_epoch_begin(self, epoch, logs=None):

    # assumes the LR for all models is the same
    for model in self.model.model_list:
      model_lr = getattr(self.model, model).optimizer.lr
      if self.initial_lr is None:
        self.initial_lr = float(backend.get_value(model_lr))
      lr = max(self.initial_lr/2**int(epoch/self.halve_after), self.min_lr)
      backend.set_value(model_lr, backend.get_value(lr))


class SAAESnapshot(tf.keras.callbacks.Callback):

  """Saves the model every certain number of iterations
  
  Attributes:
      frequency (int): frequency of snapshots in epochs
      snapshot_path (str): output path
  
  """

  def __init__(self, snapshot_path: str, frequency: int = 10):

    self.frequency = frequency
    self.snapshot_path = snapshot_path

  def on_epoch_end(self, epoch, logs=None):

    if epoch > 1 and epoch % self.frequency == 0:
      self.model.save(self.snapshot_path)
