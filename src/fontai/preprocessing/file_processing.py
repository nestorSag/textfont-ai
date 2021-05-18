from __future__ import absolute_import
from collections.abc import Iterable
import os
import logging
import string
import zipfile
import io
import typing as t
import types
from abc import ABC, abstractmethod, Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import imageio

from fontai.core import InMemoryFile, DataPath, TfrHandler, LabeledExample

logger = logging.getLogger(__name__)

class ObjectMapper(ABC):
  """
    Interface for data transformations that return a generator; useful for one-to-many transformations

  """

  @abstractmethod
  def raw_map(self,data: t.Any) -> t.Generator[t.Any, None, None]:

    """
      Applies a transformation to the input data.

    """
    pass

  def map(self,data: t.Any) -> t.Generator[t.Any, None, None]:

    """
    Processes a single data instance.

    Returns a generator with a variable number of derived data instances

    """
    output = self.raw_map(data)
    if not isinstance(output, types.GeneratorType):
      raise TypeError("Output of transform() must be a generator")
    return output


class KeyValueMapper(ObjectMapper):

  """
    Wrapper class that applies an ObjectMapper's transformation to a value in a key value pair, and carries the key over.

    mapper: An instance of an ObjectMapper's subclass
  """
  def __init__(self, mapper):

    if not isinstance(mapper, ObjectMapper):
      raise TypeError("mapper is not an instance of ObjectMapper")

    self.mapper = mapper

  def raw_map(self, pair: t.Tuple[str,t.Any]) -> t.Tuple[str,t.Any]:
    key, data = pair
    value_generator = self.mapper.map(data)
    for value in value_generator:
      yield key, value


class OneToManyMapper(ObjectMapper):

  """
    Wrapper class that applies an ObjectMapper's instance transformation to each item in a generator.

    mapper: An instance of an ObjectMapper's subclass
  """

  def __init__(self, mapper):

    if not isinstance(mapper, ObjectMapper):
      raise TypeError("mapper is not an instance of ObjectMapper")

    self.mapper = mapper

  def raw_map(self, data: Iterable[t.Any, None, None]) -> t.Generator[t.Any, None, None]:
    for elem in data:
      for derived in self.map(elem):
        yield derived


class PipelineExecutor(ObjectMapper):

  """
    Applies a sequence of transformations to an input.

    stages: a list of instances inheriting from ObjectMapper
  """

  def __init__(self, stages: t.List[ObjectMapper]):

    self.stages = stages

  def raw_map(self, data: t.Any) -> t.Generator[t.Any, None, None]:

    if not isinstance(data, Iterable):
      data = [data]

    for stage in self.stages:
      data = stage.map(data)

    for elem in data:
      yield elem


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


class DataPathReader(ObjectMapper):
  """
    Loads the bytestream from a DataPath object; returns a key-value pair needed in the Beam pipeline to persist Tensorflow records according to the source file.

  """

  def raw_map(self, path: DataPath) -> t.Generator[t.Tuple[str, InMemoryFile], None, None]:
    yield path.filename, InMemoryFile(filename = path.filename, content = path.read_bytes())



class ZipToFontFiles(ObjectMapper):

  """
    Opens an in-memory zip file and outputs individual ttf files

  """

  def raw_map(self,file: InMemoryFile)-> t.Generator[InMemoryFile, None, None]:
    logger.info(f"Extracting font files from zip file '{file.filename}'")
    with io.BytesIO(file.content) as bf:
      try:
        zipped = zipfile.ZipFile(bf)
      except Exception as e:
        logger.exception(f"Error while reading zip file '{file.filename}'")
        return
      for zipped_file in zipped.namelist():
        try:
          yield InMemoryFile(filename = zipped_file, content = zipped.read(zipped_file))
        except Exception as e:
          logger.exception(f"Error while unzipping file '{zipped_file}' from zip file '{file.filename}'")



class FontFileToCharArrays(ObjectMapper):
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

  def raw_map(self,file: InMemoryFile)-> t.Generator[LabeledExample, None, None]:
    logger.info(f"exctracting arrays from file '{file.filename}'")
    with io.BytesIO(file.content) as bf:
      try:
        font = ImageFont.truetype(bf,self.font_extraction_size)
      except Exception as e:
        logger.exception(f"Error while reading font file '{file.filename}'")
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
          yield LabeledExample(x=array,y=char,metadata=file.filename)
        except Exception as e:
          logger.exception(f"Error while reading char '{char}' from font file '{file.filename}'")


class ArrayCropper(ObjectMapper):

  """
    Crops an array and returns an array corresponding to the smallest bounding box containing all non-zero value.

  """

  def raw_map(self, example: LabeledExample) -> t.Generator[LabeledExample, None, None]:
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
      yield LabeledExample(x=cropped, y=example.y, metadata=example.metadata)


class ArrayResizer(ObjectMapper):

  """
    Resizes an image's numpy array to a square image with the specified dimensions

    output_size: height and width of output array

  """

  def __init__(self, output_size = 64):
    self.output_size = 64

  def _map(self, example: LabeledExample) -> t.Generator[LabeledExample, None, None]:
    """
    resize given image to a squared output image
    """
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
        yield LabeledExample(x=output.astype(np.uint8), y=y,metadata=metadata)
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

  def process(self,pair: t.Tuple[str, LabeledExample]) -> None:
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

