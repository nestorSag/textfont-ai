import sys
import re
import tensorflow as tf
import json

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

class SupervisedAdversarialAutoEncoder(tf.keras.Model):

  def __init__(
    self,
    encoder,
    decoder,
    discriminator,
    code_dim,
    reconstruction_loss_weight=0.5,
    input_dim=(64,64,1),
    n_classes = 62):

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

    self.prior_sampler = tf.random.normal
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

    prior_samples = self.prior_sampler(shape=(32,self.code_dim))
    with tf.GradientTape(persistent=True) as tape:
      code = self.encoder(x, training=True)
      extended_code = tf.concat([code,labels],axis=1)
      decoded = self.decoder(extended_code,training=True)  # Forward pass
      #dcdr_loss = self.decoder_loss(x,decoded)

      real = self.discriminator(prior_samples,training=True)
      fake = self.discriminator(code,training=True)
      #discr_loss = self.discriminator_loss(real,fake)

      #encoder_loss = self.rec_loss_weight*dcdr_loss + (1-self.rec_loss_weight)*discr_loss

      loss = self.rec_loss_weight*self.decoder_loss(x,decoded) + (1-self.rec_loss_weight)*self.discriminator_loss(real,fake)
      # Compute the loss value
      # (the loss function is configured in `compile()`)

    # Compute gradients
    grads = tape.gradient(loss,self.encoder.trainable_variables + self.decoder.trainable_variables + self.discriminator.trainable_variables)

    self.optimizer.apply_gradients(zip(grads,self.encoder.trainable_variables + self.decoder.trainable_variables + self.discriminator.trainable_variables))
    # discr_gradients = tape.gradient(discr_loss,self.discriminator.trainable_variables)
    # decoder_gradients = tape.gradient(dcdr_loss, self.decoder.trainable_variables)
    # encoder_gradients = tape.gradient(encoder_loss, self.encoder.trainable_variables)

    # self.discr_optimizer.apply_gradients(zip(discr_gradients,self.discriminator.trainable_variables))
    # self.decoder_optimizer.apply_gradients(zip(decoder_gradients,self.decoder.trainable_variables))
    # self.encoder_optimizer.apply_gradients(zip(encoder_gradients,self.encoder.trainable_variables))

    self.mse_metric.update_state(x,decoded)

    discr_true = tf.concat([tf.ones_like(real),tf.zeros_like(fake)],axis=0)
    discr_predicted = tf.round(tf.concat([real,fake],axis=0))
    self.accuracy_metric.update_state(discr_true,discr_predicted)

    return {"Reconstruction error": self.mse_metric.result(), "Adversarial error": self.accuracy_metric.result()}

  @property
  def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    # or at the start of `evaluate()`.
    # If you don't implement this property, you have to call
    # `reset_states()` yourself at the time of your choosing.
    return [self.mse_metric, self.accuracy_metric]

  @classmethod
  def from_config(cls,config):

    encoder = tf.keras.Sequential.from_config(config["encoder"])
    decoder = tf.keras.Sequential.from_config(config["decoder"])
    discriminator = tf.keras.Sequential.from_config(config["discriminator"])

    return cls(
      encoder,
      decoder,
      discriminator,
      config["code_dim"],
      config["reconstruction_loss_weight"],
      config["input_dim"],
      config["n_classes"])

  def get_config(self):
    d = {}
    d["encoder"] = self.encoder.get_config()
    d["decoder"] = self.decoder.get_config()
    d["discriminator"] = self.discriminator.get_config()
    d["code_dim"] = self.code_dim
    d["reconstruction_loss_weight"] = self.rec_loss_weight
    d["input_dim"] = self.input_dim
    d["n_classes"] = self.n_classes

    return d

  def save(self,output_dir):
    self.encoder.save(output_dir + "encoder")
    self.decoder.save(output_dir + "decoder")
    self.discriminator.save(output_dir + "discriminator")

    d = {
      "reconstruction_loss_weight":self.rec_loss_weight,
      "input_dim": self.input_dim,
      "n_classes": self.n_classes,
      "code_dim": self.code_dim
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



