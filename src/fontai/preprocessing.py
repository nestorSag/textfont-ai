from __future__ import absolute_import
from collections.abc import Iterable

import os
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

class InputDataHandler(object):

  def __init__(self,padding=2,pixel_threshold=100,charset="all"):

    self.record_spec = {
      'char': tf.io.FixedLenFeature([], tf.string),
      'filename': tf.io.FixedLenFeature([], tf.string),
      'img': tf.io.FixedLenFeature([], tf.string),
    }
    if charset != "all":
      if charset == "lowercase":
        self.classes = string.ascii_letters[0:26]
      elif charset == "uppercase":
        self.classes = string.ascii_letters[26::]
      elif charset == "numbers":
        self.classes = string.digits
      else:
        raise Exception("Only 'lowercase', 'uppercase', 'numbers' or 'all' are allowed as options for charset")
    else:
      self.classes = string.ascii_letters + string.digits

    self.tf_classes = tf.convert_to_tensor(list(self.classes))
    self.num_classes = len(self.classes)
    self.padding = padding
    self.pixel_threshold = pixel_threshold

  def parse_tf_objects(self,serialized):
    return tf.io.parse_single_example(serialized,self.record_spec)

  def filter_by_char(self,parsed):
    return tf.reduce_any(self.tf_classes == parsed["char"])

  def process_tf_objects(self,parsed):
    img = tf.image.decode_png(parsed["img"])
    img = tf.image.resize_with_crop_or_pad(img,target_height=64+2*self.padding,target_width=64+2*self.padding)
    img = tf.cast(img,dtype=tf.float32)
    y = tf.cast(tf.where(self.tf_classes == parsed["char"]),dtype=tf.int32)
    label = tf.reshape(tf.one_hot(indices=y,depth=self.num_classes),(self.num_classes,))#.reshape((num_classes,))
    return img, label

  def filter_sparse_images(self,img,label):
    return tf.math.count_nonzero(img) > self.pixel_threshold

  def get_dataset(self,folder):
    if folder[-1] != "/":
      folder = folder + "/"
    files = [folder + file for file in os.listdir(folder)]

    dataset = tf.data.TFRecordDataset(filenames=files)\
      .map(self.parse_tf_objects)\
      .filter(self.filter_by_char)\
      .map(self.process_tf_objects)\
      .filter(self.filter_sparse_images)

    return dataset
  def get_training_dataset(self,folder,batch_size=32):
    dataset = self.get_dataset(folder)

    dataset = dataset\
      .shuffle(buffer_size=2*batch_size)\
      .repeat()\
      .batch(batch_size)

    return dataset

  def get_evaluation_dataset(self,folder,batch_size=None):
    dataset = self.get_dataset(folder)
    
    if batch_size is not None:
      return dataset.batch(batch_size)
    else:
      return dataset