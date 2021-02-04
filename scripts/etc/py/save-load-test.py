import os
import json 
import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fontai.preprocessing import InputDataHandler
from fontai.models import * 
import tensorflow as tf

from fontai.models import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

with open("tmp/aae-encoder.json","r") as f:
  hyperparameters = json.loads(f.read())

output_size = 8
hyperparameters["layers"] = [{
  "class":"tf.keras.Input",
  "kwargs": {"shape":[64,64,1]}
}] + hyperparameters["layers"] + [{
  "class":"tf.keras.layers.Dense",
  "kwargs": {"units":output_size}
}]

encoder = get_stacked_network(hyperparameters)

# Initialise decoder
with open("tmp/aae-decoder.json","r") as f:
  hyperparameters = json.loads(f.read())

input_size = 32
hyperparameters["layers"] = [{
  "class":"tf.keras.Input",
  "kwargs": {"shape":input_size + 62}
}] + hyperparameters["layers"]

decoder = get_stacked_network(hyperparameters)

hyperparameters["layers"][-1]["activation"] = "" #this is so the dict can be saved as JSON

#Initialise discriminator
with open("tmp/aae-discriminator.json","r") as f:
  hyperparameters = json.loads(f.read())

#output_size = len(handler.classes)
hyperparameters["layers"] = [{
  "class":"tf.keras.Input",
  "kwargs": {"shape":8}
}] + hyperparameters["layers"] + [{
  "class":"tf.keras.layers.Dense",
  "kwargs": {"units":1,"activation":"sigmoid"}
}]

discriminator = get_stacked_network(hyperparameters)



class MyModel(tf.keras.Model):
  #
  def __init__(
    self,
    model1,
    model2,
    model3):
    #
    super(MyModel, self).__init__()
    self.model1 = model1
    self.model2 = model2
    self.model3 = model3
    #
  def __call__(self,x):
    return self.model2(self.model1(x))



model1 = tf.keras.Sequential(
  [tf.keras.Input(shape = (None,64,64,1)),
   tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.Dense(10,activation="softmax")
  ])


model2 = tf.keras.Sequential(
  [tf.keras.Input(shape = (None,10)),
   tf.keras.layers.Dense(20,activation="softmax")
  ])

mymodel = MyModel(encoder,decoder,discriminator)

tf.saved_model.save(mymodel,"mymodel")

mymodel = tf.keras.models.load_model("mymodel",custom_objects={"MyModel": MyModel})





model = SupervisedAdversarialAutoEncoder.load("models/aae-1/")


folder = "models/aae-1/"
encoder = tf.keras.models.load_model(folder + "encoder")
decoder = tf.keras.models.load_model(folder + "decoder")
discriminator = tf.keras.models.load_model(folder + "discriminator")