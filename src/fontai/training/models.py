import sys
import re
import io
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np

thismodule = sys.modules[__name__]


def rescaled_sigmoid_activation(factor):
  def f(x):
    return factor * tf.keras.activations.sigmoid(x)

  return f

def get_layer_instance(layer_dict):
  layer_name = layer_dict["class"]

  ## returns the layer class based on its name; looks in the tf and local modules
  try:
    if re.match("tf.keras.layers",layer_name):
      layer = getattr(tf.keras.layers,layer_name.replace("tf.keras.layers.",""))
    elif re.match("tf.keras",layer_name):
      layer = getattr(tf.keras,layer_name.replace("tf.keras.",""))
    else:
      layer = getattr(thismodule,layer_name)
    #print(f"layer: {layer}")
    return layer(**layer_dict["kwargs"])
  except Exception as e:
    raise Exception(f"an error occured instantiating a layer: {e}")

def get_stacked_network(hyperpar_dict):
  layers_dict = hyperpar_dict["layers"]
  layer_list = []
  for layer_spec in layers_dict:
    layer_list.append(get_layer_instance(layer_spec))

  model = tf.keras.Sequential(layer_list)
  #model.compile(loss = hyperpar_dict["loss"], optimizer = hyperpar_dict["optimizer"], metrics = hyperpar_dict["metrics"])
  return model

class ScatterGatherConvLayer(tf.keras.layers.Layer):
  """ This layer passes its input through multiple separate layers and then concatenate their output in a single tensor
      #with same dimensions as input
  Parameters:

  layers_info (`dict`): dict with a "submodules" key, whose value is a list of dicts that are inputs for `StackedNetwork` instances
  """

  def __init__(self,layers):
    super(ScatterGatherConvLayer,self).__init__()
    module_number = 0
    for module in layers["submodules"]:
      #layer_class = getattr(tf.keras.layers,layer["class"])
      setattr(self,"module" + str(module_number),StackedNetwork(module))
      module_number += 1
    self.module_number = module_number

  def call(self,inputs):
    return tf.concat([getattr(self,"module" + str(i))(inputs) for i in range(self.module_number)], axis=3)

# {"submodules":[{"layers":[...]},{"layers":[...]}]}


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



class SupervisedAdversarialAutoEncoder(tf.keras.Model):

  def __init__(
    self,
    encoder,
    decoder,
    discriminator,
    code_dim,
    reconstruction_loss_weight=0.5,
    input_dim=(64,64,1),
    n_classes = 62,
    prior_batch_size=32):

    super(SupervisedAdversarialAutoEncoder, self).__init__()

    #encoder.build(input_shape=input_dim)
    #decoder.build(input_shape=(None,n_classes+code_dim))
    #discriminator.build(input_shape=(None,code_dim))

    self.input_dim = input_dim
    self.n_classes = n_classes
    self.encoder = encoder
    self.decoder = decoder
    self.discriminator = discriminator
    self.code_dim = code_dim
    self.rec_loss_weight = min(max(reconstruction_loss_weight,0),1)
    self.prior_batch_size = prior_batch_size
    print(f"reconstruction loss weight: {self.rec_loss_weight}")

    self.prior_sampler = tf.random.uniform
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.mse_loss = tf.keras.losses.MSE
    self.mse_metric = tf.keras.metrics.MeanSquaredError(name="Reconstruction error")
    self.accuracy_metric = tf.keras.metrics.Accuracy(name="Adversarial error")

  def decoder_loss(self,original,decoded):
    return self.mse_loss(original,decoded)

  def discriminator_loss(self,real,fake):
    real_loss = self.cross_entropy(tf.ones_like(real), real)
    fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss)

  def __call__(self, x, training=True, mask=None):
    return self.encoder(x)

  def train_step(self, inputs):

    x, labels = inputs

    prior_samples = self.prior_sampler(shape=(self.prior_batch_size,self.code_dim),maxval=1.0)
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
      code = self.encoder(x, training=True)
      extended_code = tf.concat([code,labels],axis=1)
      decoded = self.decoder(extended_code,training=True)  # Forward pass
      #dcdr_loss = self.decoder_loss(x,decoded)

      real = self.discriminator(prior_samples,training=True)
      fake = self.discriminator(code,training=True)

      reconstruction_loss = self.decoder_loss(x,decoded)
      classification_loss = self.discriminator_loss(real,fake)
      mixed_loss = -(1-self.rec_loss_weight)*classification_loss + self.rec_loss_weight*self.decoder_loss(x,decoded)

    # Compute gradients
    
    discr_gradients = tape1.gradient(classification_loss,self.discriminator.trainable_variables)
    decoder_gradients = tape2.gradient(reconstruction_loss, self.decoder.trainable_variables)
    encoder_gradients = tape3.gradient(mixed_loss, self.encoder.trainable_variables)

    self.discriminator.optimizer.apply_gradients(zip(discr_gradients,self.discriminator.trainable_variables))
    self.decoder.optimizer.apply_gradients(zip(decoder_gradients,self.decoder.trainable_variables))
    self.encoder.optimizer.apply_gradients(zip(encoder_gradients,self.encoder.trainable_variables))

    self.mse_metric.update_state(x,decoded)

    discr_true = tf.concat([tf.ones_like(real),tf.zeros_like(fake)],axis=0)
    discr_predicted = tf.round(tf.concat([real,fake],axis=0))
    self.accuracy_metric.update_state(discr_true,discr_predicted)

    return {"MSE": self.mse_metric.result(), "Accuracy": self.accuracy_metric.result()}

  @property
  def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    # or at the start of `evaluate()`.
    # If you don't implement this property, you have to call
    # `reset_states()` yourself at the time of your choosing.
    return [self.mse_metric, self.accuracy_metric]

  def save(self,output_dir):
    self.encoder.save(output_dir + "encoder")
    self.decoder.save(output_dir + "decoder")
    self.discriminator.save(output_dir + "discriminator")

    d = {
      "reconstruction_loss_weight":self.rec_loss_weight,
      "input_dim": self.input_dim,
      "n_classes": self.n_classes,
      "code_dim": self.code_dim,
      "prior_batch_size": self.prior_batch_size
    }

    with open(output_dir + "aae-params.json","w") as f:
      json.dump(d,f)

  @classmethod
  def load(cls,folder):
    encoder = tf.keras.models.load_model(folder + "encoder")
    decoder = tf.keras.models.load_model(folder + "decoder")
    discriminator = tf.keras.models.load_model(folder + "discriminator")

    with open(folder + "aae-params.json","r") as f:
      d = json.loads(f.read())

    return SupervisedAdversarialAutoEncoder(encoder = encoder,decoder = decoder,discriminator = discriminator,**d)

