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

from fontai.preprocessing import Preprocessor

class ImageExtractor(beam.DoFn):
  
  def __init__(self, font_size = 100, font_padding=500, offset=50):
    self.font_size = font_size
    self.font_padding = font_padding
    self.offset = offset

  def process(self, gcs_file):
    #logging.info("processing {f}".format(f=gcs_file))
    try:
      zip_ = Preprocessor.get_zip_from_gcp(gcs_file)
      try:
        yield Preprocessor.extract_imgs_from_zip(zip_)
      except Exception as e:
        logging.exception("Error processing zip file {l}: {e}".format(l=gcs_file,e=e))
        return 
    except Exception as e:
      logging.exception("Error retrieving {l}: {e}".format(l=gcs_file,e=e))
      return 

class Cropper(beam.DoFn):

  def process(self,tuple_): #  -> Tuple[str,Tuple[str,np.ndarray]]
    # prepare png
    try:
      letter, output_filename, im = tuple_
      cropped = Preprocessor.crop_img(im)
      key = output_filename[0] #character in png
      return letter, output_filename, cropped
    except Exception as e:
      print("error cropping image: {e}".format(e=e))
      return 

class Resizer(beam.DoFn):

  def __init__(self,output_dim):
    self.output_dim = output_dim

  def process(self,triplet):

    try:
      letter, output_filename, img = triplet
      output = Preprocessor.resize(img,self.output_dim)
      return letter, output_filename, output
    except Exception as e:
      print(f"error resizing char {letter} in file {output_filename}: {e}")
      return 

def set_char_as_key(triplet):
  return (triplet[0],triplet)

def set_hash_as_key(triplet,n_buckets=20):
  # create hash from name
  hashed = np.random.randint(n_buckets)
  return (hashed,triplet)

class TensorCreator(beam.DoFn):
  #returns a tuple with numpy arrays to be compressed and uploaded
  def __init__(self,image_dim):
    self.image_dim = image_dim
  def process(self,kvp) -> Tuple[str,np.ndarray,np.ndarray,np.ndarray]:
    key,value_list = kvp
    #
    try:
      letters = np.array([x[0] for x in value_list])
      filenames = np.array([x[1] for x in value_list])
      imgs = np.array([x[2].reshape(self.image_dim,self.image_dim,1) for x in value_list])
      yield (key, letters, filenames, imgs)
    except Exception as e:
      print("error converting to numpy: {e}".format(e=e))

class TensorUploader(beam.DoFn):
  # uploads a compressed numpy array zip to cloud storage
  def __init__(self,output_folder):
    self.output_folder = output_folder

  def process(self,values) -> None:
    #print("values: {v}".format)
    key, chars, filenames, imgs = values

    output_suffix = ("" if self.output_folder[-1] == "/" else "/") + str(key)

    #for kind, obj in (("char",letters),("filenames",filenames),("img",imgs)):
    try:
      #save npz data
      bf = io.BytesIO()
      np.savez_compressed(bf,img=imgs,char=chars,filename=filenames)
      outfile = self.output_folder + output_suffix + ".npz"
      gcs_file = GcsIO().open(outfile,mode="w")
      gcs_file.write(bf.getvalue())
      gcs_file.close()
      bf.close()

    except Exception as e:
      logging.exception("Error uploading numpy objects for character {c}: {e}".format(c=key,e=e))


class TFRecordContentCreator(beam.DoFn):
  #returns the byte stream of zipped images to be stored in TF.Example instances
  def __init__(self,image_dim):
    self.image_dim = image_dim

  def process(self,kvp) -> Tuple[str,np.ndarray,np.ndarray,np.ndarray]:
    key,value_list = kvp

    def img_to_png_bytes(img):
      bf = io.BytesIO()
      imageio.imwrite(bf,img,"png")
      val = bf.getvalue()
      bf.close()
      return val
    #
    try:
      letters = [str.encode(x[0]) for x in value_list]
      filenames = [str.encode(x[1],errors="replace") for x in value_list]
      imgs = [img_to_png_bytes(x[2]) for x in value_list]

      yield (key, letters, filenames, imgs)
    except Exception as e:
      print("error converting to numpy: {e}".format(e=e))

      
class TFRecordUploader(beam.DoFn):
  # uploads a group of png byte arrays to cloud storage
  def __init__(self,output_folder):
    self.output_folder = output_folder

  def process(self,values) -> None:
    #print("values: {v}".format)
    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    key, chars, filenames, imgs = values

    output_suffix = ("" if self.output_folder[-1] == "/" else "/") + str(key)

    #for kind, obj in (("char",letters),("filenames",filenames),("img",imgs)):
    try:
      #save tf record
      with tf.io.TFRecordWriter(self.output_folder + output_suffix + ".tfr") as writer:
        for i in range(len(chars)):
          img = imgs[i]
          char = chars[i]
          filename = filenames[i]
          example = tf.train.Example(
            features=tf.train.Features(
              feature={
              "img": _bytes_feature(img),
              "char":_bytes_feature(bytes(char)),
              "filename":_bytes_feature(bytes(filename))}))
          writer.write(example.SerializeToString())
    except Exception as e:
      logging.exception("Error uploading numpy objects for character {c}: {e}".format(c=key,e=e))


# classes below this point were used in previous iterations to 
# output zips of PNGs grouped by character, .npy files, and .npz files
# currently the output is a TFRecord object, which are compressable and language agnostic

class DataCompressor(beam.DoFn):
  #groups png byte streams in a single zip file
  def process(self,kvp) -> Tuple[str,bytes]:
    key,value_list = kvp
    zip_bf = io.BytesIO()
    with zipfile.ZipFile(zip_bf,mode="w") as zp:
      for filename, img in value_list:
        elem_bf = io.BytesIO()
        imageio.imwrite(elem_bf,im=img,format="png")
        zp.writestr(filename,elem_bf.getvalue())
        elem_bf.close()

    zip_stream = zip_bf.getvalue()
    zip_bf.close()
    yield (key, zip_stream)


class ZipUploader(beam.DoFn):
  # upload zip of pngs to cloud storage
  def __init__(self,output_folder):
    self.output_folder = output_folder

  def process(self,kvp) -> None:
    key, byte_stream = kvp
    output_suffix = ("" if self.output_folder[-1] == "/" else "/") + key + ".zip"
    outfile = self.output_folder + output_suffix
    try:
      gcs_file = GcsIO().open(outfile,mode="w")
      gcs_file.write(byte_stream)
      gcs_file.close()
    except Exception as e:
      logging.exception("Error uploading ZIP for character {c}: {e}".format(c=key,e=e))

# class DimGatherer(beam.CombineFn):
#   # simply gather a list of image dimensions to reduce later
#   def create_accumulator(self):
#     return []

#   def add_input(self,accumulator,other):
#     if isinstance(other,Iterable) and len(other) == 2 and other[0] > 0 and other[1] > 0:
#       return accumulator + [other]
#     else:
#       return accumulator

#   def merge_accumulators(self, accumulators):
#     return [tpl for accumulator in accumulators for tpl in accumulator]

#   def extract_output(self,accumulator):
#     ## implement median calculation
#     return accumulator

# def get_median_dims(lst):
#   median = [np.quantile([x[0] for x in lst],0.5),np.quantile([x[1] for x in lst],0.5)]
#   logger.info("median is {x}".format(median))
#   return int(max(median))



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
    
    # sorted_by_hash = (standardised
    #   | 'setHashAsKey' >> beam.Map(lambda x: set_hash_as_key(x))
    #   | 'groupByHash' >> beam.GroupByKey()
    #   | 'createHashBundles' >> beam.ParDo(TensorCreator(int(user_options.png_size)))
    #   | 'saveHashToNpz' >> beam.ParDo(TensorUploader(output_folder + "sorted-by-hash")))

    sorted_by_hash = (standardised
      | 'setHashAsKey' >> beam.Map(lambda x: set_hash_as_key(x))
      | 'groupByHash' >> beam.GroupByKey()
      | 'createHashBundles' >> beam.ParDo(TFRecordContentCreator(int(user_options.png_size)))
      | 'saveHashToTFR' >> beam.ParDo(TFRecordUploader(output_folder + "sorted-by-hash")))

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()