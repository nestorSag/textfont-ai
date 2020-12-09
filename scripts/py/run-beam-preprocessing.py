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

def get_bounding_box(gray_img):
  nonzero = np.where(gray_img > 0)
  if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
    return (0, 0), (0,0)
  else:
    dims = [(np.min(axis),np.max(axis)) for axis in nonzero]
    return dims

class ImageExtractor(beam.DoFn):
  
  def __init__(self, font_size = 100, font_padding=500, offset=50):
    self.font_size = font_size
    self.font_padding = font_padding
    self.offset = offset

  def get_fontname(self,str,ext=".ttf"):
    return str.split("/")[-1].lower().replace(ext,"")

  def choose_ext(self,lst):
    ttfs = len([x for x in lst if ".ttf" in x.lower()])
    otfs = len([x for x in lst if ".otf" in x.lower()])
    if ttfs >= otfs:
      return ".ttf"
    else:
      return ".otf"

  def process(self, gcs_file):
    #logging.info("processing {f}".format(f=gcs_file))
    try:
      bf = io.BytesIO()
      bf.write(GcsIO().open(gcs_file,mode="r").read())
      zip_ = zipfile.ZipFile(bf)
      
      files_in_zip = zip_.namelist()
      # choose whether to proces TTFs or OTFs, but not both
      ext = self.choose_ext(files_in_zip)
      available = sorted([filename for filename in files_in_zip if ext in filename.lower()])
      #main_file = available[0] #pressumably the file with the shortest name is the main font type in the ZIP
      #main_fontname = self.get_fontname(main_file)

      valid_files = []
      #identifiers = []
      for filename in available:
        fontname = self.get_fontname(filename)
        try:
          valid_files.append((filename,zip_.read(filename)))
          #tag font variations for possible downstream filtering: store (filename,tag) tuples
          #tag should mostly have the form: bold, italic, 3d, etc.
          # tag = fontname.replace(main_fontname,"")
          # if tag == fontname:
          #   # if this happens, probably there is more than one font type in this ZIP, so update main_fontname
          #   # and set tag as empty
          #   main_fontname = self.get_fontname(filename)
          #   tag = ""
          # identifiers.append((filename,tag))
        except Exception as e:
          logging.exception("Error reading font file {x} from {l}: {e}".format(l=gcs_file,x=filename,e=e))
      zip_.close()
      bf.close()
    except Exception as e:
      logging.exception("error unzipping files from {f}: {e}".format(f=gcs_file,e=e))
      return 
    for name, file in valid_files:
      for letter in string.ascii_letters + string.digits:
        #logging.info("working on letter {l}".format(l=letter))
        try:
          letter_bf = io.BytesIO(file)
          im = Image.new("RGB",(self.font_padding,self.font_padding))
          draw = ImageDraw.Draw(im)
          font = ImageFont.truetype(letter_bf,self.font_size)
          #font = ImageFont.truetype(bytes(ttf),self.font_size)
          draw.text((self.offset,self.offset),letter,font=font)
          # filename indexes letter and font type, avoids overwritting data with timestamp
          output_filename = self.get_fontname(name).lower()# + str(datetime.now().time())
          letter_bf.close()
          yield letter, output_filename, im
        except Exception as e:
          logging.exception("error processing letter {x} from file {l}: {e}".format(l=gcs_file,x=letter,e=e))
          return 

class Cropper(beam.DoFn):

  def process(self,tuple_): #  -> Tuple[str,Tuple[str,np.ndarray]]
    # prepare png
    letter, output_filename, im = tuple_
    bf = io.BytesIO()
    im.save(bf,format="png")
    # get bounding box dimension
    gray_img = np.mean(imageio.imread(bf.getvalue(),format="png"),axis=-1).astype(np.uint8)
    bf.close()
    h_bound, w_bound = get_bounding_box(gray_img)
    if h_bound == (0,0) or w_bound == (0,0):
      return
    else:
      h = h_bound[1] - h_bound[0] + 1
      w = w_bound[1] - w_bound[0] + 1
      #crop and map to png
      cropped = gray_img[h_bound[0]:(h_bound[0] + h),w_bound[0]:(w_bound[0]+w)]
      key = output_filename[0] #character in png
      yield letter, output_filename, cropped

class Resizer(beam.DoFn):

  def __init__(self,output_dim):
    self.output_dim = output_dim

  def process(self,triplet):

    letter, output_filename, img = triplet
    output = np.zeros((self.output_dim,self.output_dim),dtype=np.uint8)
    # resize img to fit into output dimensions
    img_h, img_w = img.shape
    if img_h > 0 and img_w > 0:
      if img_h >= img_w:
        resize_dim = (self.output_dim,int(img_w*self.output_dim/img_h))
      else:
        resize_dim = (int(img_h*self.output_dim/img_w),self.output_dim)
      try:
        resized = np.array(Image.fromarray(np.uint8(img)).resize(size=tuple(reversed(resize_dim))))
        # embed into squared image
        resized_h, resized_w = resized.shape
        h_pad, w_pad = int((self.output_dim - resized_h)/2), int((self.output_dim - resized_w)/2)
        output[h_pad:(h_pad+resized_h),w_pad:(w_pad+resized_w)] = resized
        # make the image binary
        #output[output > int(255/2)] = 255
        #output[output < int(255/2)] = 0
        yield letter, output_filename, output.astype(np.uint8)
      except Exception as e:
        logging.exception("error resizing png: {e}".format(e=e))
        return 
    else:
      return

def set_letter_as_key(triplet):
  return (triplet[0],triplet)

def set_hash_as_key(triplet,n_buckets=20):
  # create hash from name
  hashed = str(hash(triplet[1])%n_buckets)
  return(hashed,triplet)

class TensorCreator(beam.DoFn):
  #returns the byte stream of zipped images
  def __init__(self,image_dim):
    self.image_dim = image_dim

  def process(self,kvp) -> Tuple[str,np.ndarray,np.ndarray,np.ndarray]:
    key,value_list = kvp
    #
    letters = np.array([x[0] for x in value_list])
    filenames = np.array([x[1] for x in value_list])
    imgs = np.array([x[2].reshape(self.image_dim,self.image_dim,1) for x in value_list])

    yield (key, letters, filenames, imgs)



class TensorUploader(beam.DoFn):

  def __init__(self,output_folder):
    self.output_folder = output_folder

  def process(self,values) -> None:
    #print("values: {v}".format)
    key, letters, filenames, imgs = values

    output_suffix = ("" if self.output_folder[-1] == "/" else "/") + key

    for kind, obj in (("char",letters),("filenames",filenames),("img",imgs)):
      try:
        bf = io.BytesIO()
        np.save(bf,obj)
        outfile = self.output_folder + output_suffix + "-" + kind + ".npy"
        gcs_file = GcsIO().open(outfile,mode="w")
        gcs_file.write(bf.getvalue())
        gcs_file.close()
        bf.close()
      except Exception as e:
        logging.exception("Error uploading numpy objects for character {c}: {e}".format(c=key,e=e))

class DataCompressor(beam.DoFn):
  #returns the byte stream of zipped images
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

class DimGatherer(beam.CombineFn):
  # simply gather a list of image dimensions to reduce later
  def create_accumulator(self):
    return []

  def add_input(self,accumulator,other):
    if isinstance(other,Iterable) and len(other) == 2 and other[0] > 0 and other[1] > 0:
      return accumulator + [other]
    else:
      return accumulator

  def merge_accumulators(self, accumulators):
    return [tpl for accumulator in accumulators for tpl in accumulator]

  def extract_output(self,accumulator):
    ## implement median calculation
    return accumulator

def get_median_dims(lst):
  median = [np.quantile([x[0] for x in lst],0.5),np.quantile([x[1] for x in lst],0.5)]
  logger.info("median is {x}".format(median))
  return int(max(median))

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

    # find median dimensions for entire dataset
    # dims = (cropped_pngs
    #   | 'findBoundingBox' >> beam.Map(lambda tuple: tuple[1][1].shape)
    #   | 'GetDimList' >> beam.CombineGlobally(DimGatherer())
    #   | 'findMedianDims' >> beam.Map(lambda dimlist: get_median_dims(dimlist)))

      #| 'saveBoundingBoxInfo' >> WriteToText(user_options.output_folder + ("" if user_options.output_folder[-1] == "/" else "/") + "GLOBAL_BOUNDING_BOX.txt"))

    # useful to sort by letter for generative models 
    output_folder = user_options.output_folder if user_options.output_folder[-1] == "/" else user_options.output_folder + "/"
    sorted_by_char = (standardised
      | 'setLetterAsKey' >> beam.Map(lambda x: set_letter_as_key(x))
      | 'groupByChar' >> beam.GroupByKey()
      | 'createCharTensors' >> beam.ParDo(TensorCreator(int(user_options.png_size)))
      | 'saveCharTensors' >> beam.ParDo(TensorUploader(output_folder + "sorted-by-char")))

    # useful to train letter classifiers
    sorted_by_hash = (standardised
      | 'setHashAKey' >> beam.Map(lambda x: set_hash_as_key(x))
      | 'groupByHash' >> beam.GroupByKey()
      | 'createHashTensors' >> beam.ParDo(TensorCreator(int(user_options.png_size)))
      | 'saveHashTensors' >> beam.ParDo(TensorUploader(output_folder + "sorted-by-hash")))

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()