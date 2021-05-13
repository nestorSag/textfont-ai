from __future__ import absolute_import
from collections.abc import Iterable
import os
import logging
import string
import zipfile
import io
from datetime import datetime
import typing as t
from abc import ABC

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import imageio
import tensorflow as tf

import apache_beam as beam
from apache_beam.io.gcp.gcsio import GcsIO

from fontai.config.preprocessing import Config
from fontai.core import InMemoryFile, DataPath


class ObjectMapper(ABC):
  """
    Interface for pre-ML file and data transformations

  """

  @abstractmethod
  def _map(self,data):
    pass

  def map(self,data) -> t.Generator[object]:

    """
    Processes a single data instance.

    Returns a generator with a variable number of derived data instances

    """
    output = self._map(data)
    if not isinstance(output, t.GeneratorType):
      raise TypeError("Output of transform() must be a generator")
    return output



class KeyValueMapper(ObjectMapper):
  """
    Interface for pre-ML file and data transformations; it enforces that the output of the transformation is of type KeyValuePair. This is to keep track of the zip file that originated each derived object, in order to pack them together at the end of the pipeline.

  """

  @abstractmethod
  def _map(self,data):
    pass

  def map(self,data) -> t.Generator[KeyValuePair]:

    """
    Processes a single data instance.

    Returns a generator with a variable number of derived data instances

    """
    generator = super().map(data)
    return self.wrap_output(generator)

  def wrap_output(self, generator) -> t.Generator[KeyValuePair]:
    for pair in generator:
      key,value = pair
      yield KeyValuePair(key=key,value=value)



class DataPathReader(KeyValueMapper):
  """
    Loads the bytestream from a DataPath object

  """

  def _map(self, path: DataPath):
    yield path.filename, InMemoryFile(name = path.filename, content = path.read_bytes())



class ZipToFontFiles(KeyValueMapper):

  """
    Opens an in-memory zip file and outputs individual ttf files

  """

  def _map(self,pair: KeyValuePair)-> t.Generator[InMemoryFile]:
    key, file = pair
    with io.BytesIO(file.content) as bf:
      zipped = zipfile.ZipFile(bf)
      for zipped_file in zipped.namelist():
        yield key, InMemoryFile(filename = zipped_file, content = zipped.read(zipped_file))



class FontFileToCharArrays(KeyValueMapper):
  """
    Processes ttf files and outputs labeled examples consisting of a label (character) and a numpy array corresponding to individual character images

    charset: string with all the characters to be extracted from the file

    font_size: font size used when converting them to character images

    canvas_size: image canvas size in which fonts are converted to images

    canvas_padding: size of canvas padding

  """
  def __init__(
    self, 
    charset = string.ascii_letters + string.digits, 
    font_size = 100, 
    canvas_size = 500, 
    canvas_padding = 100):

    if canvas_padding >= image_size/2:
      raise ValueError(f"Canvas padding value ({canvas_padding}) is too large for canvas size ({canvas_size})")

    self.font_size = font_size
    self.canvas_size = canvas_size
    self.canvas_padding = canvas_padding
    self.canvas_size = canvas_size
    self.charset = charset

  def _map(self,pair: KeyValuePair)-> t.Generator[np.ndarray]:
    key,file = pair
    with io.BytesIO(file.content) as bf:
      font = ImageFont.truetype(bf,self.font_size)
      for char in self.charset:
        img = Image.new("RGB",(self.canvas_size,self.canvas_size))
        draw = ImageDraw.Draw(img)
        draw.text((offset,offset),letter,font=font)

        with io.BytesIO() as bf2:
          img.save(bf2,format="png")
          array = imageio.imread(bf2.getvalue(),format="png")

        array = np.mean(array, dim = -1).astype(np.uint8)
        yield key, LabeledExample(x=array,y=char,metadata=file.name)

class ArrayCropper(KeyValueMapper):

  """
    Crops an array and returns an array corresponding to the bounding box containing all non-zero value.

  """

  def _map(self, pair: KeyValuePair) -> np.ndarray:
    key,example = pair
    nonzero = np.where(example.x > 0)
    if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
      yield key, LabeledExample(x=np.empty((0,),dtype=np.uint8), y=example.y)#(0, 0), (0,0)
    else:
      h_bound, w_bound = [(np.min(axis),np.max(axis)) for axis in nonzero]
      h = h_bound[1] - h_bound[0] + 1
      w = w_bound[1] - w_bound[0] + 1
      #crop and map to png
      cropped = example.x[h_bound[0]:(h_bound[0] + h),w_bound[0]:(w_bound[0]+w)]
      yield key, LabeledExample(x=cropped, y=example.y, metadata=example.metadata)

class ArrayResizer(KeyValueMapper):

  """
    Resizes an image's numpy array to a square image with the specified dimensions

    output_size: height and width of output array

  """

  def __init__(self, output_size = 64):
    self.output_size = 64

  def _map(self, pair: KeyValuePair):
    """
    resize given image to a squared output image
    """
    key,example = pair
    array, y, metadata = example

    output = np.zeros((self.output_size,self.output_size),dtype=np.uint8)
    # resize img to fit into output dimensions
    try:
      img_h, img_w = example.x.shape
      if img_h > 0 and img_w > 0:
        if img_h >= img_w:
          resize_dim = (self.output_size,int(img_w*self.output_size/img_h))
        else:
          resize_dim = (int(img_h*self.output_size/img_w),self.output_size)
        #try:
        resized = np.array(Image.fromarray(np.uint8(array)).resize(size=tuple(reversed(resize_dim))))
        # embed into squared image
        resized_h, resized_w = resized.shape
        h_pad, w_pad = int((self.output_size - resized_h)/2), int((self.output_size - resized_w)/2)
        output[h_pad:(h_pad+resized_h),w_pad:(w_pad+resized_w)] = resized
        # make the image binary
        yield key, LabeledExample(x=output.astype(np.uint8), y=y,metadata=metadata)
    except Exception as e:
      return



class TfrRecordWriter(ObjectMapper):
  """
    Takes a group of LabeledExamples and converts them to TF records of TF examples.

  """
  def __init__(self, output_path: DataPath):
    self.output_path = output_path

  def process(self,pair: t.Tuple[str,t.List[LabeledExample]]) -> None:
    src, examples = pair

    def img_to_png_bytes(img):
      bf = io.BytesIO()
      imageio.imwrite(bf,img,"png")
      val = bf.getvalue()
      bf.close()
      return val
    #
    full_output_path = str(self.output_path / (src + ".tfr"))
    try:
      with tf.io.TFRecordWriter(full_output_path) as writer:
        for example in examples:
          png = img_to_png_bytes(example.x)
          label = str.encode(example.y)
          metadata = str.encode(example.metadata)
          tf_example = tf.train.Example(
            features=tf.train.Features(
              feature={
              "png": _bytes_feature(img),
              "label":_bytes_feature(bytes(char)),
              "metadata":_bytes_feature(bytes(filename))}))
          writer.write(tf_example.SerializeToString())

    except Exception as e:
      logging.exception(f"error writing TF record: {e}")

      

class BeamCompatibleWrapper(beam.DoFn):

  """
    Wrapper that allows subclasses of ObjectWrapper to be used in Beam pipeline stages

    mapper: Instance of an ObjectWrapper's subclass

  """

  def __init__(self, mapper: ObjectMapper):

    if not isinstance(obj, ObjectMapper):
      raise TypeError("mapper needs to be a subclass of ObjectMapper")
    self.mapper = mapper

  def process(self, data):
    return self.mapper.map(data)




class ImageExtractor(object):
  """
  File preprocessing pipeline; takes a Config object that defines its execution.

  config: A Config instance

  """

  def __init__(self, config: Config):
    self.config = config

  def run():

    """
      Runs Beam preprocessing pipeline as defined in the config object.
    
    """
    with beam.pipelines(options=self.config.beam_parameters) as p:

      input_objs_list = self.config.input_path.list_files()

      objs = p | beam.Create(input_objs_list)# ReadFromText(input_file_list_path)

      arrays = (objs 
      | 'Load file' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = DataPathReader())
      | 'extract font files' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = ZipToFontFiles())
      | "extract arrays from font files" >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = FontFileToCharArrays(**self.config.font_to_array_config.as_dict()))
      | "crop arrays" >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = ArrayCropper())
      | "resize arrays" >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = ArrayResizer(output_size = self.config.img_output_size))
      | "convert to key value tuple" >> beam.Map(lambda pair: (pair.key,pair.value))
      | "group by key (source file)" >> beam.GroupByKey()
      | "write to disk" >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = self.TfrRecordWriter(self.config.output_path))