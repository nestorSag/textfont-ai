from __future__ import absolute_import
from collections.abc import Iterable

import argparse
import logging
import traceback
from typing import Tuple
import string
import zipfile
import io
from datetime import datetime

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import imageio

import apache_beam as beam
from apache_beam.io.gcp.gcsio import GcsIO
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.value_provider import StaticValueProvider
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

import tensorflow as tf

from src.preprocessing import * 

def run(argv=None, save_main_session=True):

  parser = argparse.ArgumentParser(description = "processes font ZIP files into individual characters' PNG files")
  parser.add_argument(
      '--input-folder',
      dest='input_folder',
      required = True,
      help='Input file to process.')
  parser.add_argument(
      '--output-folder',
      dest='output_folder',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      required = True,      
      help='Output file to write results to.')
  parser.add_argument(
      '--font-padding',
      dest='font_padding',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default='500',
      help='Dim of PNG canvas in which characters will be rendered. Needs to be large enough for provided font size')
  parser.add_argument(
      '--png-size',
      dest='png_size',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default='64',
      help='Size of fila PNG outputs')
  parser.add_argument(
      '--font-size',
      dest='font_size',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default='100',
      help='Font size to use when extracting PNGs from TTFs')
  parser.add_argument(
      '--png-offset',
      dest='png_offset',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default='128',
      help='Offset from top left corner of PNG when embedding the characters PNG; barroque fonts can overflow bounding PNG if offset is too small or too large.')

  user_options, other = parser.parse_known_args(argv)
  pipeline_options = PipelineOptions(other)

  #pipeline_options = PipelineOptions()
  #user_options = pipeline_options.view_as(UserOptions)

  if user_options.input_folder[0:5] != "gs://":
    raise Exception("Input must be a folder in GCS")

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  with beam.Pipeline(options=pipeline_options) as p:

    # these lines gather subfolder file names and create an in-memory Beam PCollection from it
    input_files = GcsIO().list_prefix(user_options.input_folder)
    input_files_list = list(input_files.keys())
    #print("input_file_list: {l}".format(l=input_files_list))

    files = p | beam.Create(input_files_list)# ReadFromText(input_file_list_path)

    # unzip files, extract character PNGs from TTFs, crop PNGs to minimal size
    standardised = (files 
    | 'getPNGs' >> beam.ParDo(ImageExtractor(
        font_size=int(user_options.font_size),
        font_padding=int(user_options.font_padding),
        offset=int(user_options.png_offset)))
    | 'cropAndGroup' >> beam.ParDo(Cropper())
    | "standardisePNGs" >> beam.ParDo(Resizer(output_dim=int(user_options.png_size))))

    # #find median dimensions for entire dataset
    # dims = (cropped_pngs
    #   | 'findBoundingBox' >> beam.Map(lambda tuple: tuple[1][1].shape)
    #   | 'GetDimList' >> beam.CombineGlobally(DimGatherer())
    #   | 'findMedianDims' >> beam.Map(lambda dimlist: get_median_dims(dimlist)))

    #   | 'saveBoundingBoxInfo' >> WriteToText(user_options.output_folder + ("" if user_options.output_folder[-1] == "/" else "/") + "GLOBAL_BOUNDING_BOX.txt"))

    
    output_folder = user_options.output_folder if user_options.output_folder[-1] == "/" else user_options.output_folder + "/"
    
    # sorted_by_char = (standardised
    #   | 'setLetterAsKey' >> beam.Map(lambda x: set_char_as_key(x))
    #   | 'groupByChar' >> beam.GroupByKey()
    #   | 'createCharBundles' >> beam.ParDo(ImageCompressor(int(user_options.png_size)))
    #   | 'saveCharsToNpz' >> beam.ParDo(TFRecordUploader(output_folder + "sorted-by-char")))

    # useful to train letter classifiers
    sorted_by_hash = (standardised
      | 'setHashAsKey' >> beam.Map(lambda x: set_hash_as_key(x))
      | 'groupByHash' >> beam.GroupByKey()
      | 'createHashBundles' >> beam.ParDo(TensorCreator(int(user_options.png_size)))
      | 'saveHashToNpz' >> beam.ParDo(TensorUploader(output_folder + "sorted-by-hash")))

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()