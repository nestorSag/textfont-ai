"""
This module contains custom Tensorflow callbacks
"""
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import mlflow

class ImageSamplerCallback(tf.keras.callbacks.Callback):

  """Generates character images from the model at the end of each epoch and pushes them to MLFLow. Shown characters are randomly chosen.
  
  """
  
  def __init__(
    self,
    n_labels: int,
    code_dim: int,
    n_imgs=16):
    """
    Args:
        n_labels (int): Number of labels in model's charset
        code_dim (int): Dimensionality of encoded representation
        n_imgs (int, optional): Number of images to sample
    
    
    """

    self.n_labels = n_labels
    self.code_dim = code_dim
    self.n_imgs = n_imgs
    self.output_file = f"tmp-{self.__class__.__name__}-output.png"

  def on_epoch_end(self,epoch,numpy_logs):

    imgs = self.generate_images()
    plot_images(imgs, self.output_file)
    mlflow.log_artifact(self.output_file)
    os.remove(self.output_file)

  def generate_images(self):

    # sample encoded representation
    samples = self.model.prior_sampler(shape=(self.n_imgs,self.code_dim))
    # sample one hot encoded labels
    labels = []
    for k in range(self.n_imgs):
      label = np.random.randint(0,self.n_labels,1)
      onehot = np.zeros((1,self.n_labels), dtype=np.float32)
      onehot[0,label] = 1.0
      labels.append(np.concatenate([samples[k],onehot],axis=-1))
    
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
    n_rows = np.ceil(n_imgs/n_cols)

    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(n_imgs):
      x = imgs[i]
      if isinstance(x, np.ndarray):
        x_ = x
      else:
        x_ = x.numpy()
      np_x = (255 * x_).astype(np.uint8).reshape((height,width))
    
    plt.savefig(output_file)

class FontSamplerCallback(ImageSamplerCallback):

  """Generates a full font from the model at the end of each epoch and pushes it to MLFLow. All characters in a given model's charset are shown in the output.
  
  """
  
  def __init__(
    self,
    n_labels: int,
    code_dim: int):
    """
    Args:
        n_labels (int): Number of labels in model's charset
        code_dim (int): Dimensionality of encoded representation
    
    """

    super().__init__(n_labels,code_dim,n_labels)

  def generate_images(self):

    # sample encoded representation
    sample = self.model.prior_sampler(shape=(1,self.code_dim))
    # sample one hot encoded labels
    labels = []
    for k in range(self.n_labels):
      onehot = np.zeros((1,self.n_labels), dtype=np.float32)
      onehot[0,k] = 1.0
      labels.append(np.concatenate([sample,onehot],axis=-1))
    
    fully_encoded = np.array(labels, dtype=np.float32)

    imgs = self.model.decoder.predict(fully_encoded)
    return imgs
