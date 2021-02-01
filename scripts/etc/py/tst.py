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


dataset = tf.data.Dataset.from_tensor_slices(["data/filtered/lowercase","data/filtered/uppercase","data/filtered/numbers"])\
    .interleave(map_func=InputDataHandler.read_gen_model_data,cycle_length=4,block_length=16)\
    .batch(100)

a,b,c = next(iter(dataset))
# charset = "lowercase"
# batch_size=1000
# # training procedure
# filter_model = tf.keras.models.load_model(f"./models/supervised-model-1-{charset}/model")

# handler = InputDataHandler(padding=0,pixel_threshold=0,charset=charset)

# dataset = handler.get_dataset(folders="data/supervised-models/train")\
#   .prefetch(2*batch_size)\
#   .batch(batch_size)\
#   .map(handler.supervised_filter(filter_model))\
#   .unbatch()


elemspec = (tf.TensorSpec(shape=(64,64,1), dtype=tf.float32, name=None), tf.TensorSpec(shape=(62,), dtype=tf.float32, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None))

dt1 = tf.data.experimental.load("data/filtered/lowercase",element_spec = elemspec_letters, compression = "GZIP")

elemspec_digits = (tf.TensorSpec(shape=(64,64,1), dtype=tf.float32, name=None), tf.TensorSpec(shape=(10,), dtype=tf.float32, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None))

dt2 = dt.concatenate(tf.data.experimental.load("data/filtered/numbers",element_spec = elemspec_digits, compression = "GZIP"))

a,b,c = next(iter(dt))

def read_curated_dataset(path,n_labels):
  elemspec = (tf.TensorSpec(shape=(64,64,1), dtype=tf.float32, name=None), tf.TensorSpec(shape=(tf.cast(n_labels,tf.int64),), dtype=tf.float32, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None))
  return tf.data.experimental.load(path,element_spec = elemspec, compression = "GZIP")

filenames = [{"file":"data/filtered/lowercase","labels":26}]

dt = tf.data.Dataset.from_tensor_slices(filenames)
dt = dt.interleave(lambda x: read_curated_dataset(x[0],x[1]),cycle_length=3,block_length=3)

dataset = tf.data.Dataset.from_tensor_slices(filenames)
def parse_fn(filename):
  return tf.data.Dataset.range(10)
dataset = dataset.interleave(lambda x:
    tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
    cycle_length=4, block_length=16)
