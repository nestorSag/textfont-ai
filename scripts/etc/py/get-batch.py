import os
import json 
import datetime
import numpy as np
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fontai.preprocessing import InputDataHandler
from fontai.models import * 
import tensorflow as tf

from fontai.models import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset = tf.data.Dataset.from_tensor_slices(["data/filtered/lowercase","data/filtered/uppercase","data/filtered/numbers"])\
  .interleave(map_func=InputDataHandler.read_gen_model_data,cycle_length=8,block_length=16)\
  .prefetch(1000)\
  .shuffle(buffer_size=2*32)\
  .repeat()\
  .batch(150)


a,b,c = next(iter(dataset))

#model = tf.saved_model.load("models/aae-1/model")

# model = tf.keras.models.load_model("models/aae-1/model",custom_objects={"SupervisedAdversarialAutoEncoder": SupervisedAdversarialAutoEncoder})

# model(a)

# f = rescaled_sigmoid_activation(255)

chars = list(string.ascii_letters + string.digits)

from PIL import Image
i = 0
for a_,b_ in zip(a,b):
  a_ = a_.numpy().reshape((64,64)).astype(np.uint8)
  im = Image.fromarray(a_)
  b_ = b_.numpy()
  char = chars[np.argmax(b_)]
  im.save(f"batch/{i}-{char}.png")
  i += 1
