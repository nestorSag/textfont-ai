import os
import argparse
import json 
import string
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from fontai.preprocessing import InputDataHandler
from fontai.evaluation import ValidationDataExaminer

from fontai.models import * 
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from PIL import Image
code_size=16
chars = list(string.ascii_letters + string.digits)

dataset = tf.data.Dataset.from_tensor_slices(["data/filtered/lowercase","data/filtered/uppercase","data/filtered/numbers"])\
  .interleave(map_func=InputDataHandler.read_gen_model_data,cycle_length=8,block_length=16)\
  .prefetch(1000)\
  .shuffle(buffer_size=2*32)\
  .repeat()\
  .batch(32)


a,b,c = next(iter(dataset))

model = SupervisedAdversarialAutoEncoder.load("models/o-aae-32/")

############
cb = SAAECallback("inspect",a,b)
cb.model = model
cb.on_epoch_end(1)
############

decoder = model.decoder
encoder = model.encoder

def get_input(char,code=None):
  if code is None:
    code = np.random.uniform(size=(code_size,))
  x = np.zeros((62,))
  x[chars.index(char)] = 1
  x = np.concatenate([code,x])
  return x.reshape((1,62+code_size))

def process_output(out):
  out = out.numpy()
  m = np.max(out)
  out = (out).astype(np.uint8).reshape((64,64))
  #out = (255.0/m*out).astype(np.uint8).reshape((64,64))
  return out

def show_output(out):
  im = Image.fromarray(out)
  im.show(f"inspect/out.png")

def get_output(char,code=None):
  input_ = get_input(char,code)
  out = decoder(input_)
  out_ = process_output(out)
  show_output(out_)


encoded = encoder(a,training=False)

sns.distplot(encoded.numpy()[0,:])

to_decode = tf.concat([encoded,b],axis=1)

decoded = decoder(to_decode)

show_output(process_output(decoded[0]))


get_output("k",encoded[0].numpy())

def sum_code(code,dim,x):
  y = code.numpy()
  y[0,dim] += x
  show_output(process_output(decoder(y)))


code = tf.reshape(tf.concat([encoded[0],b[0]],axis=0),(1,-1))