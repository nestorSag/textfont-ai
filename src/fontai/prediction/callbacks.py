"""
This module contains custom Tensorflow callbacks
"""
import sys
import re
import io
import tensorflow as tf
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fontai.io.storage import BytestreamPath
from fontai.prediction.input_processing import LabeledExamplePreprocessor 

class SAAECallback(tf.keras.callbacks.Callback):

  """Pushes diagnostic plots to Tensorboard at the end of each training epoch for a SAAE model. It takes as input a path to a Tensorflow file with a test minibatch that is to be evaluated after each epoch.
  
  Attributes:
      writer: Tensorflow file writer
      hist_step (int): Description
      output_path (str): Folder in which Tensorboard output folder resides.
      tb_folder (TYPE): Full image output path (inside Tensorboard directory)
  """
  
  def __init__(
    self,
    input_path: str, 
    output_path: str, 
    seed: int = 1, 
    batch_size: int = 32, 
    charset: str = "all"):
    """
    Args:
        input_path (str): Path to test batch; the test batch needs to be a Tensorflow records file where examples follow the same format as the training example for the SAAE model
        output_path (str): Folder in which tensorboard results are being saved
        seed (int, optional): Numpy random seed
        batch_size (int, optional): Maximum match size
        charset (str, optional): String with characters to be allowed in test minibatch
    
    """

    def load_minibatch(input_path):
        # load single minibatch from source
        loader = LabeledExamplePreprocessor(
            batch_size = batch_size,
            charset = charset,
            filters = [])

        data = loader.fetch(list(input_path.list_sources()))
        return next(iter(data))

    self.output_path = BytestreamPath(output_path)
    self.tb_folder = self.output_path / "tensorboard" / "images"

    # load test minibatch
    self.batch, self.labels = load_minibatch(BytestreamPath(input_path))

    self.writer = tf.summary.create_writer(str(self.tb_folder))
    self.hist_step: int = 0
    self.reconstruction_step: int = 0
    np.random.seed(seed)

  def on_epoch_end(self,epoch,numpy_logs):

    # produce histogram matrix of first code dimensions
    encoded = self.model.encoder(self.batch,training=False)
    hists = self.get_hists(encoded)
    with self.writer.as_default():
      tf.summary.image("Code histograms", self.plot_to_image(hists), step=self.hist_step)
    self.hist_step += 1

    a,b,c,d = self.batch.shape
    # compare input and output 
    to_decode = tf.concat([encoded,self.labels],axis=1)
    decoded = self.model.decoder(to_decode,training=False)

    rand_idx = np.random.randint(size=3,low=0,high=self.batch.shape[0],dtype=np.int64)

    with self.writer.as_default():
      for row in rand_idx:
        img = tf.concat([self.batch[row],decoded[row]],axis=1)
        img_shape = img.shape
        tf.summary.image("Input output comparison", tf.reshape(img,shape=(1,) + img_shape), step=self.reconstruction_step)
        self.reconstruction_step += 1
    # save snapshot
    self.model.save(self.output_path)

  def get_hists(self,encoded: np.ndarray,n:int=4) -> matplotlib.figure.Figure:
    """Produce histograms for each code component
    
    Args:
        encoded (np.ndarray): minibatch' encoded representation 
        n (int, optional): Maximum number of histograms to plot
    
    Returns:
        matplotlib.figure.Figure: Histograms' figure
    """

    figure = plt.figure(figsize=(10,10))
    for i in range(min(n*n,encoded.shape[1])):
      # Start next subplot.
      plt.subplot(n, n, i + 1, title=f"")
      #plt.imshow(train_images[i], cmap=plt.cm.binary)
      plt.hist(encoded[:,i].numpy())

    return figure

  def plot_to_image(self,figure: matplotlib.figure.Figure) -> tf.Tensor:
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    
    Args:
        figure (matplotlib.figure.Figure): Input figure
    
    Returns:
        tf.Tensor: Tensor representing the figure
    """


    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image