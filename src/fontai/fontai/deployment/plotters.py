import io
import base64
import typing as t

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from fontai.prediction.callbacks import SAAEFontSamplerCallback, TensorSAAEFontSamplerCallback

class AlphabetPlotter(ABC):

  @classmethod
  @abstractmethod
  def generate_font(cls):
    pass

  @classmethod
  @abstractmethod
  def plot_font(cls):
    pass

  def fig_to_str(cls, in_fig, close_all=True, **save_args):
    """Maps a pyplot figure to base64 encoding for display inside a Dash app
    
    Args:
        in_fig : Matplotlib figure
        close_all (bool, optional): Close all figures after saving to an internal buffer
        **save_args: Arguments passed when saving figure to an internal buffer
    
    Returns:
        str: base64-encoded image
    """
    out_img = io.BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return f"data:image/png;base64,{encoded}"



class SAAEAlphabetPlotter(AlphabetPlotter):

  @classmethod
  def generate_font(cls, model:tf.keras.Model, style_vector: np.array, charset_size: int):
    """Creates a set of character images as numpy arrays from a given font style vector and charset size. Assumes images are along the first axis in the resulting tensor, i.e. images are output examples.
    
    Args:
        model (tf.keras.Model): Generative model. Its input must be a vector of size charset_size + style_dim, and it should output an image as a 3-dimensional tensor
        style_vector (np.array): style vector
        charset_size (int): number of characters in font
    
    Returns:
        np.array: four-dimensional array where the first dimension correspond to image indices and the last to image channels
    """
    # sample one hot encoded labels
    labels = []
    for k in range(charset_size):
      onehot = np.zeros((1,charset_size), dtype=np.float32)
      onehot[0,k] = 1.0
      labels.append(np.concatenate([style_vector,onehot],axis=-1))
    
    fully_encoded = np.array(labels, dtype=np.float32)

    chars = model.predict(fully_encoded)
    return chars



  @classmethod
  def plot_font(cls, imgs: t.Union[tf.Tensor, np.ndarray], n_cols = 7) -> None:
    """Utility function to plot a sequence of images from numpy array in a matrix plot
    
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

    for k in range(n_rows*n_cols):
      axs[int(k/n_cols), k%n_cols].axis("off")
    
    return fig



class TensorSAAEAlphabetPlotter(AlphabetPlotter):

  @classmethod
  def generate_font(cls, model:tf.keras.Model, style_vector: np.array, *args, **kwargs):
    """Creates a set of character images as numpy arrays from a given font style vector and charset size. Assumes imags are along the last channels, i.e. images are channels in the resulting tensor. It does not pass character labels to the model as they are not needed (since they are encoded in the channel index).
    
    Args:
        model (tf.keras.Model): Generative model. Its input must be a vector of size charset_size + style_dim, and it should output an image as a 3-dimensional tensor
        style_vector (np.array): style vector
        charset_size (int): number of characters in font
    
    Returns:
        np.array: four-dimensional array where the first dimension correspond to image indices and the last to image channels
    """
    # sample one hot encoded labels
    

    chars = model.predict(style_vector.reshape((1,-1)))
    return chars

  @classmethod
  def plot_font(cls, imgs: t.Union[tf.Tensor, np.ndarray], n_cols = 7) -> None:
    """Utility function to plot a sequence of images from numpy array in a matrix plot
    
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

    for k in range(n_rows*n_cols):
      axs[int(k/n_cols), k%n_cols].axis("off")
    
    return fig

