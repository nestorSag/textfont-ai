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
batch_size = 128
padding=1
training_data_dir = "./data/train/"
output_dir = "./models/model1_lowercase"
charset = "lowercase"

# training procedure
handler = InputDataHandler(padding=padding,pixel_threshold=pixel_threshold,charset=charset)
num_classes=len(handler.classes)

dataset = handler.get_training_dataset(folder=training_data_dir,batch_size=batch_size)


with open("tmp/hyperpars.json","r") as f:
	hyperpar_dict = json.loads(f.read())

#model = get_stacked_network(hyperpar_dict)

model = tf.keras.Sequential(
  [tf.keras.Input(shape = (64+padding,64+padding,1)),
   tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.Conv2D(64,kernel_size=(8,8),activation="relu"),
   tf.keras.layers.Conv2D(96,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
   tf.keras.layers.Conv2D(96,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.Conv2D(128,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.Conv2D(160,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
   tf.keras.layers.Conv2D(192,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.Conv2D(224,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.Conv2D(256,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.Conv2D(64,kernel_size=(1,1),activation="relu"),
   tf.keras.layers.Flatten(),
   #tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(num_classes,activation="softmax")
  ]
)

model.summary()

model.compile(loss = hyperpar_dict["loss"], optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), metrics = hyperpar_dict["metrics"])


log_dir = "logs/fit/1x1conv"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#/etc/modprobe.d/nvidia-kernel-common.conf

model.fit(dataset, steps_per_epoch = int(1000000*len(handler.classes)/62/batch_size), epochs = 8, callbacks=[tensorboard_callback])

model.save(output_dir)