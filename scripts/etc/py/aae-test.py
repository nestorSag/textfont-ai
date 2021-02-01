import os
import json 
import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fontai.preprocessing import InputDataHandler
from fontai.models import * 
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# parameters
pixel_threshold = 150
batch_size = 32
padding=0
training_data_dir = "./data/supervised-models/train/"
output_dir = "./models/model1_lowercase"
charset = "lowercase"
n_epochs = 2
code_size = 8
img_shape = 64 + 2*padding
# training procedure
handler = InputDataHandler(padding=padding,pixel_threshold=pixel_threshold,charset=charset)
# x,y=handler.find_sample_statistics("data/train/2.tfr")
filter_model = tf.keras.models.load_model(f"./models/supervised-model-1-{charset}/model")

dataset = handler.get_dataset(folder=training_data_dir)\
  .filter(handler.supervised_filter(filter_model))

dataset = handler.scramble_dataset(dataset).prefetch(batch_size*4)

with open("tmp/aae-encoder.json","r") as f:
  hyperparameters = json.loads(f.read())

#output_size = len(handler.classes)
output_size = code_size
hyperparameters["layers"] = [{
  "class":"tf.keras.Input",
  "kwargs": {"shape":[img_shape,img_shape,1]}
}] + hyperparameters["layers"] + [{
  "class":"tf.keras.layers.Dense",
  "kwargs": {"units":output_size}
}]

encoder = get_stacked_network(hyperparameters)

with open("tmp/aae-decoder.json","r") as f:
  hyperparameters = json.loads(f.read())

input_size = code_size
#output_size = len(handler.classes)
hyperparameters["layers"] = [{
  "class":"tf.keras.Input",
  "kwargs": {"shape":code_size + len(handler.classes)}
}] + hyperparameters["layers"]

decoder = get_stacked_network(hyperparameters)

with open("tmp/aae-discriminator.json","r") as f:
  hyperparameters = json.loads(f.read())

#output_size = len(handler.classes)
hyperparameters["layers"] = [{
  "class":"tf.keras.Input",
  "kwargs": {"shape":code_size}
}] + hyperparameters["layers"] + [{
  "class":"tf.keras.layers.Dense",
  "kwargs": {"units":1,"activation":"sigmoid"}
}]

discriminator = get_stacked_network(hyperparameters)

# batch, labels = next(iter(dataset))

# encoded = encoder(batch)
# decoded = decoder(tf.concat([encoded,labels],axis=1))
# discr = discriminator(encoded)

model = AdversarialAutoEncoder(
  encoder=encoder,
  decoder=decoder,
  discriminator=discriminator,
  discriminator_input_dim=code_size)

model.compile(loss = hyperparameters["loss"], optimizer = "adam", metrics = hyperparameters["metrics"])

model.fit(dataset, steps_per_epoch = 50, epochs = n_epochs)