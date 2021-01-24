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
padding=1
training_data_dir = "./data/train/"
output_dir = "./models/model1_lowercase"
charset = "lowercase"
n_epochs = 2

# training procedure
handler = InputDataHandler(padding=padding,pixel_threshold=pixel_threshold,charset=charset)
# x,y=handler.find_sample_statistics("data/train/2.tfr")
num_classes=len(handler.classes)

dataset = handler.get_training_dataset(folder=training_data_dir,batch_size=batch_size)


with open("tmp/classifier-64-3x3-padded-leakyrelu-batchnorm.json","r") as f:
	hyperparameters = json.loads(f.read())

input_size = 64 + 2*padding
output_size = len(handler.classes)
hyperparameters["layers"] = [{
  "class":"tf.keras.Input",
  "kwargs": {"shape":[input_size,input_size,1]}
}] + hyperparameters["layers"] + [{
  "class":"tf.keras.layers.Dense",
  "kwargs": {"units":output_size,"activation":"softmax"}
}]

model = get_stacked_network(hyperparameters)
optimizer = getattr(tf.keras.optimizers,hyperparameters["optimizer"]["class"])(**hyperparameters["optimizer"]["kwargs"])

model.compile(loss = hyperparameters["loss"], optimizer = optimizer, metrics = hyperparameters["metrics"])

model.summary()

#model.compile(loss = hyperparameters["loss"], optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), metrics = hyperparameters["metrics"])

# log_dir = "logs/fit/1x1conv"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#/etc/modprobe.d/nvidia-kernel-common.conf

#model.fit(dataset, steps_per_epoch = 200, epochs = 2)
model.fit(dataset, steps_per_epoch = 5000, epochs = n_epochs)

model.save(output_dir)

# early stopping: keep track of validation loss

# data augmentation: rotation, reflexion, distortion, random cropping, random noise
# RMSProp: scale gradient by an exponentially decayed average of the gradient norm: normalise stochastic gradient observations
# ADAM: gradient descent with momentum, plus RMSProp-style normalisation: normlaised stochastic observations of momentum SGD 
# dropout makes sense on fully connected layers: sort of an ensemble of models