import os
import json 
import datetime
import argparse
import numpy as np
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fontai.preprocessing import InputDataHandler
from fontai.models import * 
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run(argv):
  parser = argparse.ArgumentParser(description = "Filter image data using trained models; keep only correctly classified images (removes quircky fonts and images with errors in metadata).")
  parser.add_argument(
      '--output-folder',
      dest='output_folder',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      required = True,      
      help='Output folder where Tensorflow datasets will be saved')
  parser.add_argument(
      '--input-folder',
      dest='input_folders',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      required = True,
      action="append",      
      help='Input folder; can be provided multiple times to process multiple folders')
  parser.add_argument(
      '--charset',
      dest='charset',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      required = True,      
      help='character set used ("lowercase","uppercase" or "numbers")')
  parser.add_argument(
      '--filter-model',
      dest='filter_model',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      required = True,      
      help='Saved tf model that will be used to filter images.')
  parser.add_argument(
      '--batch-size',
      dest='batch_size',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      required = False,
      default = 128,
      type=int,   
      help='Saved tf model that will be used to filter images.')
  parser.add_argument(
      '--n-shards',
      dest='n_shards',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default = 40,   
      type=int,
      help='Saved tf model that will be used to filter images.')
  #elemspec = (tf.TensorSpec(shape=(64,64,1), dtype=tf.float32, name=None), tf.TensorSpec(shape=(26,), dtype=tf.float32, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None))

  args, _ = parser.parse_known_args(argv)

  def shard(img,label,filename):
    return np.random.randint(low=1,high=args.n_shards,dtype=np.int64)

  handler = InputDataHandler(padding=0,pixel_threshold=0,charset=args.charset)
  filter_model = tf.keras.models.load_model(args.filter_model)
  print(f"input folders: {args.input_folders}")
  dataset = handler.get_dataset(folders=args.input_folders)\
    .prefetch(2*args.batch_size)\
    .batch(args.batch_size)\
    .map(handler.supervised_filter(filter_model))\
    .unbatch()\
    .map(handler.expand_labels)
  #outpath = f"./data/generative-models/{charset}"
  Path(args.output_folder).mkdir(parents=True, exist_ok=True)
  tf.data.experimental.save(dataset,path=args.output_folder,compression="GZIP",shard_func=shard)


if __name__ == "__main__":
  run(sys.argv)
