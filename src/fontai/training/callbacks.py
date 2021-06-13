import sys
import re
import io
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np

class SAAECallback(tf.keras.callbacks.Callback):

  def __init__(self,output_dir,batch,labels,seed=1):

    self.output_dir = output_dir + ("/" if output_dir[-1] != "/" else "")
    self.tb_folder = self.output_dir + "tensorboard/images"
    self.batch = batch
    self.labels = labels
    self.file_writer = tf.summary.create_file_writer(self.tb_folder)
    self.hist_step = 0
    self.reconstruction_step = 0
    np.random.seed(seed)

  def on_epoch_end(self,epoch,numpy_logs):

    # produce histogram matrix of first code dimensions
    encoded = self.model.encoder(self.batch,training=False)
    hists = self.get_hists(encoded)
    with self.file_writer.as_default():
      tf.summary.image("Code histograms", self.plot_to_image(hists), step=self.hist_step)
    self.hist_step += 1

    a,b,c,d = self.batch.shape
    # compare input and output 
    to_decode = tf.concat([encoded,self.labels],axis=1)
    decoded = self.model.decoder(to_decode,training=False)

    rand_idx = np.random.randint(size=3,low=0,high=self.batch.shape[0],dtype=np.int64)

    with self.file_writer.as_default():
      for row in rand_idx:
        img = tf.concat([self.batch[row],decoded[row]],axis=1)
        img_shape = img.shape
        tf.summary.image("Input output comparison", tf.reshape(img,shape=(1,) + img_shape), step=self.reconstruction_step)
        self.reconstruction_step += 1
    # save snapshot
    self.model.save(self.output_dir)

  def get_hists(self,encoded,n=4):
    figure = plt.figure(figsize=(10,10))
    for i in range(min(n*n,encoded.shape[1])):
      # Start next subplot.
      plt.subplot(n, n, i + 1, title=f"")
      #plt.imshow(train_images[i], cmap=plt.cm.binary)
      plt.hist(encoded[:,i].numpy())

    return figure

  def plot_to_image(self,figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
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