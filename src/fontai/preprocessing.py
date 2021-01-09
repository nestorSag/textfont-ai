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
import tensorflow as tf

class Preprocessor(object):

  @staticmethod
  def get_bounding_box(gray_img):
    """
    Takes an image and returns coordinates for smallest bounding box for nonzero elements

    Returns:
    (h_low,h_high), (w_low,w_high)
    """
    nonzero = np.where(gray_img > 0)
    if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
      return (0, 0), (0,0)
    else:
      dims = [(np.min(axis),np.max(axis)) for axis in nonzero]
      return dims

  @staticmethod
  def get_filename(str,ext=".ttf"):
    """
    Extract filename from file path, assumed to be font name
    """
    return str.split("/")[-1].lower().replace(ext,"")

  @staticmethod
  def choose_ext(lst):
    """
    choose an extension to work on based on the extension's frequency in a list of zipped files.
    """
    ttfs = len([x for x in lst if ".ttf" in x.lower()])
    otfs = len([x for x in lst if ".otf" in x.lower()])
    if ttfs >= otfs:
      return ".ttf"
    else:
      return ".otf"

  @staticmethod
  def extract_imgs_from_zip(zip_,canvas_size,font_size,offset):
    """
    Generator that extracts multiple images as numpy arrays (one per character) from a zip file of font files

    Returns:
    triplet with character, filename and numpy array
    """
    files_in_zip = zip_.namelist()
    # choose whether to proces TTFs or OTFs, but not both
    ext = Preprocessor.choose_ext(files_in_zip)
    available = sorted([filename for filename in files_in_zip if ext in filename.lower()])
    valid_files = []
    #identifiers = []
    for filename in available:
      fontname = Preprocessor.get_filename(filename)
      try:
        valid_files.append((filename,zip_.read(filename)))
      except Exception as e:
        logging.exception("Error reading font file {x}: {e}".format(x=filename,e=e))
    for name, file in valid_files:
      for letter in string.ascii_letters + string.digits:
        #logging.info("working on letter {l}".format(l=letter))
        try:
          letter_bf = io.BytesIO(file)
          im = Image.new("RGB",(canvas_size,canvas_size))
          draw = ImageDraw.Draw(im)
          font = ImageFont.truetype(letter_bf,font_size)
          #font = ImageFont.truetype(bytes(ttf),self.font_size)
          draw.text((offset,offset),letter,font=font)
          # filename indexes letter and font type, avoids overwritting data with timestamp
          output_filename = Preprocessor.get_filename(name).lower()# + str(datetime.now().time())
          letter_bf.close()
          yield letter, output_filename, im
        except Exception as e:
          logging.exception("error processing letter {x} from file {l}: {e}".format(l=file,x=letter,e=e))
          return 

  @staticmethod
  def crop_img(im):
    """
    Crops numpy array according to smallest bounding box for non-zero elements
    """
    bf = io.BytesIO()
    im.save(bf,format="png")
    # get bounding box dimension
    gray_img = np.mean(imageio.imread(bf.getvalue(),format="png"),axis=-1).astype(np.uint8)
    bf.close()
    h_bound, w_bound = Preprocessor.get_bounding_box(gray_img)
    if h_bound == (0,0) or w_bound == (0,0):
      return
    else:
      h = h_bound[1] - h_bound[0] + 1
      w = w_bound[1] - w_bound[0] + 1
      #crop and map to png
      cropped = gray_img[h_bound[0]:(h_bound[0] + h),w_bound[0]:(w_bound[0]+w)]
      return cropped

  @staticmethod
  def resize(img,output_dim):
    """
    resize given image to a squared output image
    """
    output = np.zeros((output_dim,output_dim),dtype=np.uint8)
    # resize img to fit into output dimensions
    img_h, img_w = img.shape
    if img_h > 0 and img_w > 0:
      if img_h >= img_w:
        resize_dim = (output_dim,int(img_w*output_dim/img_h))
      else:
        resize_dim = (int(img_h*output_dim/img_w),output_dim)
      #try:
      resized = np.array(Image.fromarray(np.uint8(img)).resize(size=tuple(reversed(resize_dim))))
      # embed into squared image
      resized_h, resized_w = resized.shape
      h_pad, w_pad = int((output_dim - resized_h)/2), int((output_dim - resized_w)/2)
      output[h_pad:(h_pad+resized_h),w_pad:(w_pad+resized_w)] = resized
      # make the image binary
      #output[output > int(255/2)] = 255
      #output[output < int(255/2)] = 0
      return output.astype(np.uint8)
      # except Exception as e:
      #   logging.exception("error resizing png: {e}".format(e=e))
      #   return 

class TFRHandler(object):

  def __init__(self):

    self.record_spec = {
      'char': tf.io.FixedLenFeature([], tf.string),
      'filename': tf.io.FixedLenFeature([], tf.string),
      'img': tf.io.FixedLenFeature([], tf.string),
    }

    self.classes = string.ascii_letters + string.digits

  def parse_record(self,serialized):
    return tf.io.parse_single_example(serialized,self.record_spec)


  def to_numpy(self,filepath):
    records = tf.data.TFRecordDataset(filepath)
    examples = records.map(self.parse_record)

    imgs = []
    filenames = []
    chars = []
    for example in examples:
      img = imageio.imread(io.BytesIO(example["img"].numpy()))
      imgs.append(img.reshape((1,) + img.shape + (1,)))
      filenames.append(example["filename"].numpy().decode("utf-8"))
      chars.append(example["char"].numpy().decode("utf-8"))

    imgs = np.concatenate(imgs,axis=0)
    return imgs.astype(np.int32), np.array([self.classes.index(char) for char in chars],dtype=np.int32), np.array(filenames), np.array(chars)
