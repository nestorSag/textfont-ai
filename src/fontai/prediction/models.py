import sys
import re
import io
import tensorflow as tf
import json
import logging
import copy
import matplotlib.pyplot as plt
import numpy as np

from fontai.io.storage import BytestreamPath

logger = logging.getLogger(__name__)

class CharStyleSAAE(tf.keras.Model):

  """This class fits a supervised adversarial autoencoder as laid out in "Adversarial Autoencoders" by Ian Goodfellow et al.
  
  Attributes:
      accuracy_metric (tf.keras.metrics.Accuracy): Accuracy metric
      cross_entropy (tf.keras.losses.BinaryCrossentropy): Cross entropy loss
      decoder (tf.keras.Model): Decoder model
      prior_discriminator (tf.keras.Model): prior_discriminator model
      full_encoder (tf.keras.Model): Encoder that takes high-level image features and labels
      image_encoder (tf.keras.Model): Encoder for image features
      input_dim (t.Tuple[int]): Input dimension
      mse_loss (TYPE): Description
      mse_metric (tf.keras.losses.MSE): MSE loss
      prior_batch_size (int): Batch size from prior distribution at training time
      prior_sampler : Object from which prior samples are generated
      rec_loss_weight (float): Weight of reconstruction loss at training time. Should be between 0 and 1.
  """

  prior_sampler = tf.random.normal
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) #assumes prior_discriminator outputs probabilities(i.e. in [0, 1])

  mse_metric = tf.keras.metrics.MeanSquaredError(name="Reconstruction error")
  prior_accuracy_metric = tf.keras.metrics.Accuracy(name="prior adversarial accuracy")
  cross_entropy_metric = tf.keras.metrics.BinaryCrossentropy(name="Prior adversarial cross entropy", from_logits=False)

  def __init__(
    self,
    full_encoder: tf.keras.Model,
    image_encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    prior_discriminator: tf.keras.Model,
    reconstruction_loss_weight:float=0.5,
    prior_batch_size:int=32):
    """Summary
    
    Args:
        decoder (tf.keras.Model): Decoder model
        prior_discriminator (tf.keras.Model): prior_discriminator model
        full_encoder (tf.keras.Model): Encoder that takes high-level image features and labels
        image_encoder (tf.keras.Model): Encoder for image features
        reconstruction_loss_weight (float, optional): Weight of reconstruction loss at training time. Should be between 0 and 1.
        n_classes (int): number of labeled classes
        prior_batch_size (int): Batch size from prior distribution at training time
    """
    super(SAAE, self).__init__()

    #encoder.build(input_shape=input_dim)
    #decoder.build(input_shape=(None,n_classes+code_dim))
    #prior_discriminator.build(input_shape=(None,code_dim))

    self.full_encoder = full_encoder
    self.image_encoder = image_encoder
    self.decoder = decoder
    self.prior_discriminator = prior_discriminator
    self.rec_loss_weight = min(max(reconstruction_loss_weight,0),1)
    self.prior_batch_size = prior_batch_size
  
  def compile(self,
    optimizer='rmsprop',
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    **kwargs):

    self.full_encoder.compile(optimizer = copy.deepcopy(optimizer))
    self.image_encoder.compile(optimizer = copy.deepcopy(optimizer))
    self.decoder.compile(optimizer = copy.deepcopy(optimizer))
    self.prior_discriminator.compile(optimizer = copy.deepcopy(optimizer))

    super().compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      loss_weights=loss_weights,
      weighted_metrics=weighted_metrics,
      run_eagerly=run_eagerly,
      **kwargs)

  def prior_discriminator_loss(self,real,fake):
    real_loss = self.cross_entropy(tf.ones_like(real), real)
    fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss)/(2*self.prior_batch_size)

  def __call__(self, x, training=True, mask=None):
    return self.encoder(x, training=training)

  def train_step(self, inputs):

    x, labels = inputs

    #self.prior_batch_size = x.shape[0]
    #logger.info("prior_batch_size is deprecated; setting it equal to batch size.")

    with tf.GradientTape(persistent=True) as tape:

      # apply autoencoder
      image_precode = self.image_encoder(x, training=True)
      full_precode = tf.concat([image_precode, labels], axis=-1)
      code = self.full_encoder(full_precode, training=True)
      extended_code = tf.concat([code,labels],axis=-1)
      decoded = self.decoder(extended_code,training=True)  

      # apply prior_discriminator model
      prior_samples = self.prior_sampler(shape=(self.prior_batch_size,code.shape[1]))
      real = self.prior_discriminator(prior_samples,training=True)
      fake = self.prior_discriminator(code,training=True)

      # compute losses for the models
      reconstruction_loss = self.mse_loss(x,decoded)
      classification_loss = self.prior_discriminator_loss(real,fake)
      mixed_loss = -(1-self.rec_loss_weight)*classification_loss + self.rec_loss_weight*reconstruction_loss

    # Compute gradients
    discr_gradients = tape.gradient(classification_loss,self.prior_discriminator.trainable_variables)
    decoder_gradients = tape.gradient(reconstruction_loss, self.decoder.trainable_variables)
    image_encoder_gradients = tape.gradient(mixed_loss, self.image_encoder.trainable_variables)
    full_encoder_gradients = tape.gradient(mixed_loss, self.full_encoder.trainable_variables)


    #apply gradients
    self.prior_discriminator.optimizer.apply_gradients(zip(discr_gradients,self.prior_discriminator.trainable_variables))
    self.decoder.optimizer.apply_gradients(zip(decoder_gradients,self.decoder.trainable_variables))
    self.image_encoder.optimizer.apply_gradients(zip(image_encoder_gradients,self.image_encoder.trainable_variables))
    self.full_encoder.optimizer.apply_gradients(zip(full_encoder_gradients,self.full_encoder.trainable_variables))

    # compute metrics
    self.mse_metric.update_state(x,decoded)

    discr_true = tf.concat([tf.ones_like(real),tf.zeros_like(fake)],axis=0)
    discr_predicted = tf.concat([real,fake],axis=0)
    self.prior_accuracy_metric.update_state(discr_true,tf.round(discr_predicted))

    #self.cross_entropy_metric.update_state(discr_true, discr_predicted)

    return {"MSE": self.mse_metric.result(), "Prior accuracy": self.prior_accuracy_metric.result()}

  @property
  def metrics(self):
    """Performance metrics to report at training time
    
    Returns: A list of metric objects

    """
    return [self.mse_metric, self.prior_accuracy_metric, self.cross_entropy_metric]

  def predict(self, *args, **kwargs):
    return self.encoder.predict(*args,**kwargs)


  def save(self,output_dir: str):
    """Save the model to an output folder
    
    Args:
        output_dir (str): Target output folder
    """
    self.full_encoder.save(str(BytestreamPath(output_dir) / "full_encoder"))
    self.image_encoder.save(str(BytestreamPath(output_dir) / "image_encoder"))
    self.decoder.save(str(BytestreamPath(output_dir) / "decoder"))
    self.prior_discriminator.save(str(BytestreamPath(output_dir) / "prior_discriminator"))

    d = {
      "reconstruction_loss_weight":self.rec_loss_weight,
      "prior_batch_size": self.prior_batch_size
    }

    with open(str(BytestreamPath(output_dir) / "aae-params.json"),"w") as f:
      json.dump(d,f)

  @classmethod
  def load(cls, input_dir: str):
    """Loads a saved instance of this class
    
    Args:
        input_dir (str): Target input folder
    
    Returns:
        SAAE: Loaded model
    """
    full_encoder = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "full_encoder"))
    image_encoder = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "image_encoder"))
    decoder = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "decoder"))
    prior_discriminator = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "prior_discriminator"))

    with open(str(BytestreamPath(input_dir) / "aae-params.json"),"r") as f:
      d = json.loads(f.read())

    return cls(
      image_encoder = image_encoder, 
      full_encoder = full_encoder, 
      decoder = decoder, 
      prior_discriminator = prior_discriminator, 
      **d)



class PureCharStyleSAAE(CharStyleSAAE):

  """This model is trained as a regular SAAE bur an additional prior_discriminator model is added to ensure the embedding does not retain information about the character class; i.e. it only retains style information
  """

  style_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  style_accuracy_metric = tf.keras.metrics.Accuracy(name="style-char accuracy")

  def __init__(
    self,
    char_discriminator: tf.keras.Model,
    *args,
    **kwargs):

    self.char_discriminator = char_discriminator
    super().__init__(*args, **kwargs)

  def compile(self,
    optimizer=tf.keras.optimizers.Optimizer,
    *args,
    **kwargs):

    self.char_discriminator.compile(optimizer = copy.deepcopy(optimizer))
    super().compile(*args, **kwargs)


  def train_step(self, inputs):

    x, labels = inputs

    #self.prior_batch_size = x.shape[0]
    #logger.info("prior_batch_size is deprecated; setting it equal to batch size.")

    with tf.GradientTape(persistent=True) as tape:

      # apply autoencoder
      image_precode = self.image_encoder(x, training=True)
      full_precode = tf.concat([image_precode, labels], axis=-1)
      code = self.full_encoder(full_precode, training=True)
      extended_code = tf.concat([code,labels],axis=-1)
      decoded = self.decoder(extended_code,training=True)  

      # apply prior_discriminator model
      prior_samples = self.prior_sampler(shape=(self.prior_batch_size,code.shape[1]))
      real = self.prior_discriminator(prior_samples,training=True)
      fake = self.prior_discriminator(code,training=True)

      # apply char_discriminator model
      char_guess = self.char_discriminator(code,training=True)

      # compute losses for the models
      char_info_leak = self.multi_cross_entropy(labels, char_guess)/self.prior_batch_size
      reconstruction_loss = self.mse_loss(x,decoded)
      classification_loss = self.prior_discriminator_loss(real,fake)

      mixed_loss = -(1-self.rec_loss_weight)*(classification_loss + char_info_leak) + self.rec_loss_weight*(reconstruction_loss)

    # Compute gradients
    discr_gradients = tape.gradient(classification_loss,self.prior_discriminator.trainable_variables)
    decoder_gradients = tape.gradient(reconstruction_loss, self.decoder.trainable_variables)
    image_encoder_gradients = tape.gradient(mixed_loss, self.image_encoder.trainable_variables)
    full_encoder_gradients = tape.gradient(mixed_loss, self.full_encoder.trainable_variables)
    style_discr_gradients = tape.gradients(char_info_leak, self.char_discriminator.trainable_variables)


    #apply gradients
    self.prior_discriminator.optimizer.apply_gradients(zip(discr_gradients,self.prior_discriminator.trainable_variables))
    self.decoder.optimizer.apply_gradients(zip(decoder_gradients,self.decoder.trainable_variables))
    self.image_encoder.optimizer.apply_gradients(zip(image_encoder_gradients,self.image_encoder.trainable_variables))
    self.full_encoder.optimizer.apply_gradients(zip(full_encoder_gradients,self.full_encoder.trainable_variables))

    self.char_discriminator.optimizer.apply_gradients(
      zip(style_discr_gradients, self.char_discriminator.trainable_variables))
    
    # compute metrics
    self.mse_metric.update_state(x,decoded)

    discr_true = tf.concat([tf.ones_like(real),tf.zeros_like(fake)],axis=0)
    discr_predicted = tf.concat([real,fake],axis=0)
    self.accuracy_metric.update_state(discr_true,tf.round(discr_predicted))

    self.cross_entropy_metric.update_state(discr_true, discr_predicted)

    self.mean_metric.update_state(code_variance)

    self.style_accuracy_metric.update_state(tf.argmax(labels, axis=-1), tf.argmax(char_guess, axis=-1))

    return {
    "MSE": self.mse_metric.result(), 
    "Prior accuracy": self.accuracy_metric.result(), 
    "Style accuracy": self.style_accuracy_metric.result()}

  def save(self,output_dir: str):
    """Save the model to an output folder
    
    Args:
        output_dir (str): Target output folder
    """
    self.char_discriminator.save(str(BytestreamPath(output_dir) / "char_discriminator"))
    super().save(output_dir)

  @classmethod
  def load(cls, input_dir: str):
    """Loads a saved instance of this class
    
    Args:
        input_dir (str): Target input folder
    
    Returns:
        SAAE: Loaded model
    """
    char_discriminator = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "char_discriminator"))
    full_encoder = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "full_encoder"))
    image_encoder = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "image_encoder"))
    decoder = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "decoder"))
    prior_discriminator = tf.keras.models.load_model(str(BytestreamPath(input_dir) / "prior_discriminator"))

    with open(str(BytestreamPath(input_dir) / "aae-params.json"),"r") as f:
      d = json.loads(f.read())

    return cls(
      char_discriminator = char_discriminator,
      image_encoder = image_encoder, 
      full_encoder = full_encoder, 
      decoder = decoder, 
      prior_discriminator = prior_discriminator, 
      **d)





class PureFontStyleSAAE(PureCharStyleSAAE):

  """This model is trained on all characters from a single font file at a time; the aim is to encode the font's style as opossed to single characters' styles, which can happen when training with scrambled characters from different fonts and results in sometimes having different-looking fonts for a single point in latent space. This model penalizes the variance in latent representation from the same font, so ideally any character in a given font maps to a single point in latent space.
  """
  mean_metric = tf.keras.metrics.Mean(name="Mean code variance")

  def train_step(self, inputs):

    x, labels = inputs

    #self.prior_batch_size = x.shape[0]
    #logger.info("prior_batch_size is deprecated; setting it equal to batch size.")

    with tf.GradientTape(persistent=True) as tape:

      # apply autoencoder
      image_precode = self.image_encoder(x, training=True)
      full_precode = tf.concat([image_precode, labels], axis=-1)
      code = self.full_encoder(full_precode, training=True)
      extended_code = tf.concat([code,labels],axis=-1)
      decoded = self.decoder(extended_code,training=True)  

      # apply prior_discriminator model
      prior_samples = self.prior_sampler(shape=(self.prior_batch_size,code.shape[1]))
      real = self.prior_discriminator(prior_samples,training=True)
      fake = self.prior_discriminator(code,training=True)

      # apply char_discriminator model
      char_guess = self.char_discriminator(code,training=True)

      # compute losses for the models
      char_info_leak = self.multi_cross_entropy(labels, char_guess)/self.prior_batch_size
      reconstruction_loss = self.mse_loss(x,decoded)
      classification_loss = self.prior_discriminator_loss(real,fake)
      code_mean = tf.stop_gradient(tf.math.reduce_mean(code, axis=-1))
      variance_loss = tf.reduce_mean((code - code_mean)**2)

      mixed_loss = -(1-self.rec_loss_weight)*(classification_loss + char_info_leak) + self.rec_loss_weight*(reconstruction_loss + variance_loss)

    # Compute gradients
    discr_gradients = tape.gradient(classification_loss,self.prior_discriminator.trainable_variables)
    decoder_gradients = tape.gradient(reconstruction_loss, self.decoder.trainable_variables)
    image_encoder_gradients = tape.gradient(mixed_loss, self.image_encoder.trainable_variables)
    full_encoder_gradients = tape.gradient(mixed_loss, self.full_encoder.trainable_variables)
    style_discr_gradients = tape.gradients(char_info_leak, self.char_discriminator.trainable_variables)


    # Compute gradients
    discr_gradients = tape.gradient(classification_loss,self.prior_discriminator.trainable_variables)
    decoder_gradients = tape.gradient(reconstruction_loss, self.decoder.trainable_variables)
    image_encoder_gradients = tape.gradient(mixed_loss, self.image_encoder.trainable_variables)
    full_encoder_gradients = tape.gradient(mixed_loss, self.full_encoder.trainable_variables)


    #apply gradients
    self.prior_discriminator.optimizer.apply_gradients(zip(discr_gradients,self.prior_discriminator.trainable_variables))
    self.decoder.optimizer.apply_gradients(zip(decoder_gradients,self.decoder.trainable_variables))
    self.image_encoder.optimizer.apply_gradients(zip(image_encoder_gradients,self.image_encoder.trainable_variables))
    self.full_encoder.optimizer.apply_gradients(zip(full_encoder_gradients,self.full_encoder.trainable_variables))
    self.char_discriminator.optimizer.apply_gradients(
      zip(style_discr_gradients, self.char_discriminator.trainable_variables))

    # compute metrics
    self.mse_metric.update_state(x,decoded)

    discr_true = tf.concat([tf.ones_like(real),tf.zeros_like(fake)],axis=0)
    discr_predicted = tf.concat([real,fake],axis=0)
    self.accuracy_metric.update_state(discr_true,tf.round(discr_predicted))

    #self.cross_entropy_metric.update_state(discr_true, discr_predicted)

    self.mean_metric.update_state(variance_loss)

    self.style_accuracy_metric.update_state(tf.argmax(labels, axis=-1), tf.argmax(char_guess, axis=-1))

    return {
    "MSE": self.mse_metric.result(), 
    "Prior accuracy": self.accuracy_metric.result(), 
    "code_variance": self.mean_metric.result(),
    "style_accuracy": self.style_accuracy_metric.result()}


