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

  """This class fits a supervised adversarial autoencoder and its inspired in the architecture from "Adversarial Autoencoders" by Ian Goodfellow et al. The only difference is that label (i.e. character) information is injected between the encoder and the style code, in the hope that labels not only help the decoding but also the encoding process, e.g. curvyness shouldn't be as important in the input if its a C than if its an H.
  
  Attributes:
      accuracy_metric (tf.keras.metrics.Accuracy): Accuracy metric
      cross_entropy (tf.keras.losses.BinaryCrossentropy): Cross entropy loss
      decoder (tf.keras.Model): Decoder model that maps style and characters to images
      prior_discriminator (tf.keras.Model): Discriminator between the embeddings' distribution and the target distribution, e.g. multivariate standard normal.
      full_encoder (tf.keras.Model): Encoder that takes high-level image features and labels to produce embedded representations
      image_encoder (tf.keras.Model): Encoder for image features
      input_dim (t.Tuple[int]): Input dimension
      mse_loss (TYPE): Description
      mse_metric (tf.keras.losses.MSE): MSE loss
      prior_batch_size (int): Batch size from prior distribution at training time
      rec_loss_weight (float): Weight of reconstruction loss at training time. Should be between 0 and 1.
  """

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
        decoder (tf.keras.Model): Decoder model that maps style and characters to images
        prior_discriminator (tf.keras.Model): Discriminator between the embeddings' distribution and the target distribution, e.g. multivariate standard normal.
        full_encoder (tf.keras.Model): Encoder that takes high-level image features and labels to produce embedded representations
        image_encoder (tf.keras.Model): Encoder for image features
        reconstruction_loss_weight (float, optional): Weight of reconstruction loss at training time. Should be between 0 and 1.
        n_classes (int): number of labeled classes
        prior_batch_size (int): Batch size from prior distribution at training time
    """
    super(CharStyleSAAE, self).__init__()

    #encoder.build(input_shape=input_dim)
    #decoder.build(input_shape=(None,n_classes+code_dim))
    #prior_discriminator.build(input_shape=(None,code_dim))

    self.full_encoder = full_encoder
    self.image_encoder = image_encoder
    self.decoder = decoder
    self.prior_discriminator = prior_discriminator
    self.rec_loss_weight = min(max(reconstruction_loss_weight,0),1)
    self.prior_batch_size = prior_batch_size

    self.prior_sampler = tf.random.normal
    
    # list of embedded models as instance attributes 
    self.model_list = ["full_encoder", "image_encoder", "decoder", "prior_discriminator"]

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
    return self.image_encoder(x, training=training)

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
      #print((self.prior_batch_size,code.shape[1]))
      prior_samples = self.prior_sampler(shape=[self.prior_batch_size,code.shape[1]])
      real = self.prior_discriminator(prior_samples,training=True)
      fake = self.prior_discriminator(code,training=True)

      # compute losses for the models
      reconstruction_loss = tf.keras.losses.MSE(x,decoded)
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
    return self.image_encoder.predict(*args,**kwargs)


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

    (BytestreamPath(output_dir) / "aae-params.json").write_bytes(json.dumps(d).encode())

    # with open(str(BytestreamPath(output_dir) / "aae-params.json"),"w") as f:
    #   json.dump(d,f)

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

    d_string = (BytestreamPath(input_dir) / "aae-params.json").read_bytes().decode("utf-8")
    d = json.loads(d_string)

    # with open(str(BytestreamPath(input_dir) / "aae-params.json"),"r") as f:
    #   d = json.loads(f.read())

    return cls(
      image_encoder = image_encoder, 
      full_encoder = full_encoder, 
      decoder = decoder, 
      prior_discriminator = prior_discriminator, 
      **d)


















class PureCharStyleSAAE(tf.keras.Model):

  """This model is trained as a regular SAAE but an additional discriminator model is added to ensure the embedding does not retain information about the character class; i.e. it only retains style information
  """
  mean_metric = tf.keras.metrics.Mean(name="Mean code variance")
  char_accuracy_metric = tf.keras.metrics.Accuracy(name="style-char accuracy")
  prior_accuracy_metric = tf.keras.metrics.Accuracy(name="prior adversarial accuracy")
  mse_metric = tf.keras.metrics.MeanSquaredError(name="Reconstruction error")
  cross_entropy_metric = tf.keras.metrics.BinaryCrossentropy(name="Prior adversarial cross entropy", from_logits=False)


  style_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) #assumes prior_discriminator outputs 

    ## this __init__ has to be copy pasted from the model above, because Tensorflow hates good coding practices aparently
  def __init__(
    self,
    full_encoder: tf.keras.Model,
    image_encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    char_discriminator: tf.keras.Model,
    prior_discriminator: tf.keras.Model,
    reconstruction_loss_weight:float=0.5,
    prior_batch_size:int=32):
    """Summary
    
    Args:
        decoder (tf.keras.Model): Decoder model that maps style and characters to images
        prior_discriminator (tf.keras.Model): Discriminator between the embeddings' distribution and the target distribution, e.g. multivariate standard normal.
        full_encoder (tf.keras.Model): Encoder that takes high-level image features and labels to produce embedded representations
        char_discriminator (tf.keras.Model): Discriminator to remove any character information from embeddings
        image_encoder (tf.keras.Model): Encoder for image features
        reconstruction_loss_weight (float, optional): Weight of reconstruction loss at training time. Should be between 0 and 1.
        n_classes (int): number of labeled classes
        prior_batch_size (int): Batch size from prior distribution at training time
    """
    super(PureCharStyleSAAE, self).__init__()

    self.full_encoder = full_encoder
    self.image_encoder = image_encoder
    self.decoder = decoder
    self.prior_discriminator = prior_discriminator
    self.rec_loss_weight = min(max(reconstruction_loss_weight,0),1)
    self.prior_batch_size = prior_batch_size
    self.char_discriminator = char_discriminator

    self.prior_sampler = tf.random.normal
    # list of embedded models as instance attributes 
    self.model_list = ["full_encoder", "image_encoder", "decoder", "prior_discriminator", "char_discriminator"]

  def prior_discriminator_loss(self,real,fake):
    real_loss = self.cross_entropy(tf.ones_like(real), real)
    fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss)/(2*self.prior_batch_size)

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
    self.char_discriminator.compile(optimizer=copy.deepcopy(optimizer))

    super().compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      loss_weights=loss_weights,
      weighted_metrics=weighted_metrics,
      run_eagerly=run_eagerly,
      **kwargs)

  def __call__(self, x, training=True, mask=None):
    return self.image_encoder(x, training=training)

  def train_step(self, inputs):
    #prior_sampler = tf.random.normal
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
      char_info_leak = self.style_loss(labels, char_guess)/self.prior_batch_size
      reconstruction_loss = tf.keras.losses.MSE(x,decoded)
      classification_loss = self.prior_discriminator_loss(real,fake)

      mixed_loss = -(1-self.rec_loss_weight)*(classification_loss + char_info_leak) + self.rec_loss_weight*(reconstruction_loss)

    # Compute gradients
    discr_gradients = tape.gradient(classification_loss,self.prior_discriminator.trainable_variables)
    decoder_gradients = tape.gradient(reconstruction_loss, self.decoder.trainable_variables)
    image_encoder_gradients = tape.gradient(mixed_loss, self.image_encoder.trainable_variables)
    full_encoder_gradients = tape.gradient(mixed_loss, self.full_encoder.trainable_variables)
    style_discr_gradients = tape.gradient(char_info_leak, self.char_discriminator.trainable_variables)

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
    self.prior_accuracy_metric.update_state(discr_true,tf.round(discr_predicted))

    #self.cross_entropy_metric.update_state(discr_true, discr_predicted)


    self.char_accuracy_metric.update_state(tf.argmax(labels, axis=-1), tf.argmax(char_guess, axis=-1))

    return {
    "MSE": self.mse_metric.result(), 
    "Prior accuracy": self.prior_accuracy_metric.result(), 
    "style_accuracy": self.char_accuracy_metric.result()}



  def save(self,output_dir: str):
    """Save the model to an output folder
    
    Args:
        output_dir (str): Target output folder
    """

    self.char_discriminator.save(str(BytestreamPath(output_dir) / "char_discriminator"))
    self.full_encoder.save(str(BytestreamPath(output_dir) / "full_encoder"))
    self.image_encoder.save(str(BytestreamPath(output_dir) / "image_encoder"))
    self.decoder.save(str(BytestreamPath(output_dir) / "decoder"))
    self.prior_discriminator.save(str(BytestreamPath(output_dir) / "prior_discriminator"))

    d = {
      "reconstruction_loss_weight":self.rec_loss_weight,
      "prior_batch_size": self.prior_batch_size
    }

    # with open(str(BytestreamPath(output_dir) / "aae-params.json"),"w") as f:
    #   json.dump(d,f)
    (BytestreamPath(output_dir) / "aae-params.json").write_bytes(json.dumps(d).encode())

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

    # with open(str(BytestreamPath(input_dir) / "aae-params.json"),"r") as f:
    #   d = json.loads(f.read())

    d_string = (BytestreamPath(input_dir) / "aae-params.json").read_bytes().decode("utf-8")
    d = json.loads(d_string)

    return cls(
      image_encoder = image_encoder, 
      full_encoder = full_encoder, 
      decoder = decoder, 
      prior_discriminator = prior_discriminator, 
      char_discriminator = char_discriminator,
      **d)







class PureFontStyleSAAE(tf.keras.Model):

  """This model is trained on all characters from one or more font files at a time; the aim is to encode the font's style as opposed to single characters' styles, which can happen when training with scrambled characters from different fonts and results in sometimes having different-looking image styles for a given style in latent space. This model works with characters from a single typeface at a time, and use the style from a given character to decode a different randomly chosen character in the same font, using the encoded style and target label information. 
  """
  prior_accuracy_metric = tf.keras.metrics.Accuracy(name="prior adversarial accuracy")
  mse_metric = tf.keras.metrics.MeanSquaredError(name="Reconstruction error")
  cross_entropy_metric = tf.keras.metrics.BinaryCrossentropy(name="Prior adversarial cross entropy", from_logits=False)

  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) #assumes prior_discriminator outputs 

    ## this __init__ has to be copy pasted from the model above, because Tensorflow hates good coding practices aparently
  def __init__(
    self,
    full_encoder: tf.keras.Model,
    image_encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    prior_discriminator: tf.keras.Model,
    reconstruction_loss_weight:float=0.5,
    prior_batch_size:int=32,
    code_regularisation_weight=0):
    """Summary
    
    Args:
        decoder (tf.keras.Model): Decoder model that maps style and characters to images
        prior_discriminator (tf.keras.Model): Discriminator between the embeddings' distribution and the target distribution, e.g. multivariate standard normal.
        full_encoder (tf.keras.Model): Encoder that takes high-level image features and labels to produce embedded representations
        image_encoder (tf.keras.Model): Encoder for image features
        reconstruction_loss_weight (float, optional): Weight of reconstruction loss at training time. Should be between 0 and 1.
        n_classes (int): number of labeled classes
        prior_batch_size (int): Batch size from prior distribution at training time
    """
    super(PureFontStyleSAAE, self).__init__()

    self.full_encoder = full_encoder
    self.image_encoder = image_encoder
    self.decoder = decoder
    self.prior_discriminator = prior_discriminator
    self.rec_loss_weight = min(max(reconstruction_loss_weight,0),1)
    self.prior_batch_size = prior_batch_size

    self.prior_sampler = tf.random.normal
    self.code_regularisation_weight = code_regularisation_weight
    # list of embedded models as instance attributes 
    self.model_list = ["full_encoder", "image_encoder", "decoder", "prior_discriminator"]


  def prior_discriminator_loss(self,real,fake):
    real_loss = self.cross_entropy(tf.ones_like(real), real)
    fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss)/(2*self.prior_batch_size)

  def compile(self,
    optimizer='rmsprop',
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    **kwargs):

    self.full_encoder.compile(optimizer = copy.deepcopy(optimizer),run_eagerly=run_eagerly)
    self.image_encoder.compile(optimizer = copy.deepcopy(optimizer),run_eagerly=run_eagerly)
    self.decoder.compile(optimizer = copy.deepcopy(optimizer),run_eagerly=run_eagerly)
    self.prior_discriminator.compile(optimizer = copy.deepcopy(optimizer),run_eagerly=run_eagerly)  

    super().compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      loss_weights=loss_weights,
      weighted_metrics=weighted_metrics,
      run_eagerly=run_eagerly,
      **kwargs)

  def __call__(self, x, training=True, mask=None):
    return self.image_encoder(x, training=training)

  def scramble_font_batches(self, x: tf.Tensor, labels: tf.Tensor):
    """Creates a scrambled copy of a minibatch in which individual fonts are randomly shuffled. Returns the original minibatch in addition to the shuffled version.
    
    Args:
        x (tf.Tensor): Feature tensor
        labels (tf.Tensor): Label tensor
    
    Returns:
        t.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: return scrambled and original feature-label pairs, in that order.
    """
    #
    x_shape = tf.shape(x)
    n_fonts = x_shape[0]
    #
    style_x = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    style_y = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    outcome_x = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    outcome_y = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    #
    for k in range(n_fonts):
      x_k, labels_k = x[k], labels[k]
      x_k_shape = tf.shape(x_k)
      shuffling_idx = tf.range(start=0, limit=tf.shape(x_k)[0], dtype=tf.int32)
      scrambled = tf.random.shuffle(shuffling_idx)
      #
      style_x = style_x.write(k, tf.gather(x_k, scrambled, axis=0))
      style_y = style_y.write(k, tf.gather(labels_k, scrambled, axis=0))

      outcome_x = outcome_x.write(k, x_k)
      outcome_y = outcome_y.write(k, labels_k)
      #
    #
    #
    return style_x.concat(), style_y.concat(), outcome_x.concat(), outcome_y.concat()

  def train_step(self, inputs):
    #prior_sampler = tf.random.normal
    x, labels = inputs

    #self.prior_batch_size = x.shape[0]
    style_x, style_y, outcome_x, outcome_y = self.scramble_font_batches(x ,labels)

    n_examples = tf.shape(style_x)[0]
    with tf.GradientTape(persistent=True) as tape:

      # apply autoencoder
      image_precode = self.image_encoder(style_x, training=True)
      full_precode = tf.concat([image_precode, style_y], axis=-1)
      code = self.full_encoder(full_precode, training=True)
      extended_code = tf.concat([code,outcome_y],axis=-1)
      decoded = self.decoder(extended_code,training=True)  

      # apply prior_discriminator model
      prior_samples = self.prior_sampler(shape=(n_examples,code.shape[1]))
      real = self.prior_discriminator(prior_samples,training=True)
      fake = self.prior_discriminator(code,training=True)


      # Moment regularization for embedded representation to keep it closer to standard normal
      reg = self.code_regularisation_weight*(tf.reduce_mean(code)**2 + (tf.reduce_mean(code**2) - 1.0)**2)/2

      # compute losses for the models
      reconstruction_loss = tf.keras.losses.MSE(outcome_x,decoded) 
      classification_loss = self.prior_discriminator_loss(real,fake)

      mixed_loss = -(1-self.rec_loss_weight)*(classification_loss) + self.rec_loss_weight*(reconstruction_loss + reg)


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
    self.mse_metric.update_state(outcome_x,decoded)

    discr_true = tf.concat([tf.ones_like(real),tf.zeros_like(fake)],axis=0)
    discr_predicted = tf.concat([real,fake],axis=0)
    self.prior_accuracy_metric.update_state(discr_true,tf.round(discr_predicted))

    return {
    "MSE": self.mse_metric.result(), 
    "Prior accuracy": self.prior_accuracy_metric.result()}



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
      "prior_batch_size": self.prior_batch_size,
      "code_regularisation_weight": self.code_regularisation_weight
    }

    # with open(str(BytestreamPath(output_dir) / "aae-params.json"),"w") as f:
    #   json.dump(d,f)

    (BytestreamPath(output_dir) / "aae-params.json").write_bytes(json.dumps(d).encode())

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

    # with open(str(BytestreamPath(input_dir) / "aae-params.json"),"r") as f:
    #   d = json.loads(f.read())
    d_string = (BytestreamPath(input_dir) / "aae-params.json").read_bytes().decode("utf-8")
    d = json.loads(d_string)

    return cls(
      image_encoder = image_encoder, 
      full_encoder = full_encoder, 
      decoder = decoder, 
      prior_discriminator = prior_discriminator, 
      **d)


