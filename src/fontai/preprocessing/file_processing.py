from __future__ import absolute_import
from collections.abc import Iterable
import os
import logging
import string
import zipfile
import io
import typing as t
import types
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import imageio
import tensorflow as tf

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io.gcp.gcsio import GcsIO

from fontai.config.preprocessing import Config
from fontai.core import InMemoryFile, DataPath, TfrHandler, KeyValuePair, LabeledExample

logger = logging.getLogger(__name__)

class ObjectMapper(ABC):
  """
    Interface for pre-ML file and data transformations

  """

  @abstractmethod
  def _map(self,data: t.Any) -> t.Generator[t.Any, None, None]:
    pass

  def map(self,data: t.Any) -> t.Generator[t.Any, None, None]:

    """
    Processes a single data instance.

    Returns a generator with a variable number of derived data instances

    """
    output = self._map(data)
    if not isinstance(output, types.GeneratorType):
      raise TypeError("Output of transform() must be a generator")
    return output



class KeyValueMapper(ObjectMapper):
  """
    Interface for pre-ML file and data transformations; it enforces that the output of the transformation is of type KeyValuePair. This is to keep track of the zip file that originated each derived object, in order to pack them together at the end of the pipeline.

  """

  @abstractmethod
  def _map(self,data: t.Any) -> t.Generator[t.Tuple[str,t.Union[InMemoryFile,LabeledExample]], None, None]:
    pass

  def map(self,data: t.Any) -> t.Generator[KeyValuePair, None, None]:

    """
    Processes a single data instance.

    Returns a generator with a variable number of derived data instances

    """
    generator = super().map(data)
    return self.wrap_output(generator)

  def wrap_output(self, generator: t.Generator[t.Tuple[str,t.Union[InMemoryFile,LabeledExample]], None, None]) -> t.Generator[KeyValuePair, None, None]:
    for pair in generator:
      key,value = pair
      yield KeyValuePair(key=key,value=value)



class DataPathReader(KeyValueMapper):
  """
    Loads the bytestream from a DataPath object

  """

  def _map(self, path: DataPath) -> t.Generator[t.Tuple[str, InMemoryFile], None, None]:
    yield path.filename, InMemoryFile(filename = path.filename, content = path.read_bytes())



class ZipToFontFiles(KeyValueMapper):

  """
    Opens an in-memory zip file and outputs individual ttf files

  """

  def _map(self,pair: KeyValuePair)-> t.Generator[t.Tuple[str, InMemoryFile], None, None]:
    key, file = pair
    logger.info(f"Extracting font files from zip file '{file.filename}'")
    with io.BytesIO(file.content) as bf:
      try:
        zipped = zipfile.ZipFile(bf)
      except Exception as e:
        logger.exception(f"Error while reading zip file '{file.filename}'")
        return
      for zipped_file in zipped.namelist():
        try:
          yield key, InMemoryFile(filename = zipped_file, content = zipped.read(zipped_file))
        except Exception as e:
          logger.exception(f"Error while unzipping file '{zipped_file}' from zip file '{file.filename}'")




class FontFileToCharArrays(KeyValueMapper):
  """
    Processes ttf files and outputs labeled examples consisting of a label (character), a numpy array corresponding to individual character images and a metadata string indicating the original filename

    charset: string with all the characters to be extracted from the file

    font_extraction_size: font size used when converting them to character images

    canvas_size: image canvas size in which fonts are converted to images

    canvas_padding: size of canvas padding

  """
  def __init__(
    self, 
    charset = string.ascii_letters + string.digits, 
    font_extraction_size = 100, 
    canvas_size = 500, 
    canvas_padding = 100):

    if canvas_padding >= canvas_size/2:
      raise ValueError(f"Canvas padding value ({canvas_padding}) is too large for canvas size ({canvas_size})")

    self.font_extraction_size = font_extraction_size
    self.canvas_size = canvas_size
    self.canvas_padding = canvas_padding
    self.canvas_size = canvas_size
    self.charset = charset

  def _map(self,pair: KeyValuePair)-> t.Generator[t.Tuple[str, LabeledExample], None, None]:
    key,file = pair
    logger.info(f"exctracting arrays from file '{file.filename}', source file '{key}'")
    with io.BytesIO(file.content) as bf:
      try:
        font = ImageFont.truetype(bf,self.font_extraction_size)
      except Exception as e:
        logger.exception(f"Error while reading font file '{file.filename}', source file '{key}'")
        return
      for char in self.charset:
        img = Image.new("RGB",(self.canvas_size,self.canvas_size))
        draw = ImageDraw.Draw(img)
        try:
          draw.text((self.canvas_padding,self.canvas_padding),char,font=font)
          with io.BytesIO() as bf2:
            img.save(bf2,format="png")
            array = imageio.imread(bf2.getvalue(),format="png")

          array = np.mean(array, axis = -1).astype(np.uint8)
          yield key, LabeledExample(x=array,y=char,metadata=file.filename)
        except Exception as e:
          logger.exception(f"Error while reading char '{char}' from font file '{file.filename}', source file '{key}'")

class ArrayCropper(KeyValueMapper):

  """
    Crops an array and returns an array corresponding to the smallest bounding box containing all non-zero value.

  """

  def _map(self, pair: KeyValuePair) -> t.Generator[t.Tuple[str, LabeledExample], None, None]:
    key,example = pair
    nonzero = np.where(example.x > 0)
    if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
      logger.info("Empty image found. ignoring.")
      return
      #yield key, LabeledExample(x=np.empty((0,),dtype=np.uint8), y=example.y)#(0, 0), (0,0)
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

  def _map(self, pair: KeyValuePair) -> t.Generator[t.Tuple[str, LabeledExample], None, None]:
    """
    resize given image to a squared output image
    """
    key,example = pair
    array, y, metadata = example

    output = np.zeros((self.output_size,self.output_size),dtype=np.uint8)
    # resize img to fit into output dimensions
    try:
      height, width = example.x.shape
      if height > 0 and width > 0:
        if height >= width:
          resize_dim = (self.output_size,int(width*self.output_size/height))
        else:
          resize_dim = (int(height*self.output_size/width),self.output_size)
        #try:
        resized = np.array(Image.fromarray(np.uint8(array)).resize(size=tuple(reversed(resize_dim))))
        # embed into squared image
        resized_h, resized_w = resized.shape
        h_pad, w_pad = int((self.output_size - resized_h)/2), int((self.output_size - resized_w)/2)
        output[h_pad:(h_pad+resized_h),w_pad:(w_pad+resized_w)] = resized
        # make the image binary
        yield key, LabeledExample(x=output.astype(np.uint8), y=y,metadata=metadata)
    except Exception as e:
      logger.exception(f"Error while resizing array: {e}")
      return



class TfrRecordWriter(beam.DoFn):
  """
    Takes instances of LabeledExamples and writes them to a tensorflow record file.

  """
  def __init__(self, output_path: DataPath):
    self.output_path = output_path
    self.tfr_handler = TfrHandler()
    self.filename = None
    self.writer = None
    self.written_files = []

  def img_to_png_bytes(self, img):
    bf = io.BytesIO()
    imageio.imwrite(bf,img,"png")
    val = bf.getvalue()
    bf.close()
    return val

  def format_tfr_contents(self, example: LabeledExample) -> t.Tuple[bytes,bytes,bytes]:
    png = self.img_to_png_bytes(example.x)
    label = str.encode(example.y)
    metadata = str.encode(example.metadata)

    return png, label, metadata

  def process(self,pair: KeyValuePair) -> None:
    src, example = pair

    if self.filename is None:
      self.open_writer(src)
    elif self.filename != src:
      self.close()
      self.open_writer(src)
      #raise ValueError(f"key ({src}) does not match output file name {self.filename}.tfr")
    try:
      tf_example = self.tfr_handler.as_tfr(*self.format_tfr_contents(example))
      self.writer.write(tf_example.SerializeToString())

    except Exception as e:
      logging.exception(f"error writing TF record for key {src}: {e}")

  def open_writer(self, filename):
    if filename in self.written_files:
      raise ValueError(f"filename ({filename}) has already been used. Data would be lost by overwriting it.")
    self.filename = filename
    self.full_output_path = self.output_path / (self.filename + ".tfr")
    self.writer = tf.io.TFRecordWriter(str(self.full_output_path))

  def close(self):
    if self.writer is not None:
      self.writer.close()
      self.written_files.append(self.filename)

  def teardown(self):
    self.writer.close()

class BeamCompatibleWrapper(beam.DoFn):

  """
    Wrapper that allows subclasses of ObjectMapper to be used in Beam pipeline stages

    mapper: Instance of an ObjectMapper's subclass

  """

  def __init__(self, mapper: ObjectMapper):

    if not isinstance(mapper, ObjectMapper):
      raise TypeError("mapper needs to be a subclass of ObjectMapper")
    self.mapper = mapper

  def process(self, data):
    return self.mapper.map(data)




class FileProcessor(object):
  """
  File preprocessing pipeline that maps zipped font files to Tensorflow records for ML consumption; takes a Config object that defines its execution.

  config: A Config instance

  """

  def __init__(self, config: Config):
    self.config = config

  def run(self):

    """
      Runs Beam preprocessing pipeline as defined in the config object.
    
    """

    # if output is locally persisted, create parent folders
    if not self.config.output_path.is_gcs:
      Path(str(self.config.output_path)).mkdir(parents=True, exist_ok=True)

    pipeline_options = PipelineOptions(self.config.beam_cmd_line_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:

      input_objs_list = self.config.input_path.list_files()

      source_stream = p | beam.Create(input_objs_list)# ReadFromText(input_file_list_path)

      # execute pipeline
      (source_stream 
      | 'Load file' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = DataPathReader()))
      | 'extract font files' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = ZipToFontFiles()))
      | "extract arrays from font files" >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = FontFileToCharArrays(**self.config.font_to_array_config.as_dict())))
      | "crop arrays" >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = ArrayCropper()))
      | "resize arrays" >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = ArrayResizer(output_size = self.config.output_array_size)))
      | "write to disk" >> beam.ParDo(TfrRecordWriter(self.config.output_path)))

      #output_writer.close()

      # | "convert to key value tuple" >> beam.Map(lambda pair: (pair.key,pair.value))
      # | "group by key (source file)" >> beam.GroupByKey()
      # | "write to disk" >> beam.ParDo(TfrRecordWriter(self.config.output_path)))