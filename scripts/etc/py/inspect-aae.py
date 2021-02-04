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
code_size=32
chars = list(string.ascii_letters + string.digits)

dataset = tf.data.Dataset.from_tensor_slices(["data/filtered/lowercase","data/filtered/uppercase","data/filtered/numbers"])\
  .interleave(map_func=InputDataHandler.read_gen_model_data,cycle_length=8,block_length=16)\
  .prefetch(1000)\
  .shuffle(buffer_size=2*32)\
  .repeat()\
  .batch(32)


a,b,c = next(iter(dataset))

model = SupervisedAdversarialAutoEncoder.load("models/aae-cs-32-dec-1024-4096-enc-2048/")

############
cb = SAAECallback("inspect",a,b)
cb.model = model
cb.on_epoch_end(1)
############

decoder = model.decoder
encoder = model.encoder

def get_input(char):
  x = np.zeros((62,))
  x[chars.index(char)] = 1
  x = np.concatenate([np.random.normal(size=(code_size,)),x])
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

def get_output(char):
  input_ = get_input(char)
  out = decoder(input_)
  out_ = process_output(out)
  show_output(out_)

get_output(char="k")

encoded = encoder(a,training=False)

sns.distplot(encoded.numpy()[0,:])

to_decode = tf.concat([encoded,b],axis=1)

decoded = decoder(to_decode)

show_output(process_output(decoded[0]))