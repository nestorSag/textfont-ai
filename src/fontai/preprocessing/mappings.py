"""This module contains all the transformations and abstractions required to extract labeled examples ready to be used for ML training from zipped font files.

"""
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
import apache_beam as beam


from fontai.io.formats import InMemoryZipHolder, InMemoryFontfileHolder, InMemoryFile
from fontai.io.writers import BatchWriter, TfrWriter
from fontai.io.storage import BytestreamPath
from fontai.io.records import LabeledChar, LabeledFont


logger = logging.getLogger(__name__)

class ObjectMapper(ABC):


  """
  Interface for data transformations that return a generator; useful for one-to-many transformations
  
  """

  @abstractmethod
  def raw_map(self,data: t.Any) -> t.Generator[t.Any, None, None]:

    """
    Applies a transformation to the input data.
    
    Args:
        data (t.Any): input data
    
    """
    pass

  def map(self,data: t.Any) -> t.Generator[t.Any, None, None]:

    """
    Processes a single data instance and returns a generator with output data
        
    Args:
        data (t.Any): input data
    
    Returns:
        t.Generator[t.Any, None, None]: A generator with a variable number of derived data instances
    
    Raises:
        TypeError: Raised if the transformation implemented in raw_map does not return a generator
    
    """
    output = self.raw_map(data)
    if not isinstance(output, types.GeneratorType):
      raise TypeError("Output of transform() must be a generator")
    return output


class ManyToManyMapper(ObjectMapper):

  """
  Wrapper Wrapper class to apply trnsformations to an entire generator of input data
  
  
  Attributes:
      mapper (ObjectMapper): Description
  """

  def __init__(self, mapper):

    if not isinstance(mapper, ObjectMapper):
      raise TypeError("mapper is not an instance of ObjectMapper")

    self.mapper = mapper

  def raw_map(self, data: t.Iterable[t.Any]) -> t.Generator[t.Any, None, None]:
    for elem in data:
      for derived in self.mapper.map(elem):
        yield derived


class PipelineExecutor(ObjectMapper):

  """
  Applies a sequence of transformations to input data.
    
  Attributes:
      stages (ObjectMapper): A list of instances inheriting from ObjectMapper
  """

  def __init__(self, stages: t.List[ObjectMapper]):

    self.stages = stages
  def raw_map(self, data: t.Any) -> t.Generator[t.Any, None, None]:

    for stage in self.stages:
      data = stage.map(data)

    return data


class BeamCompatibleWrapper(beam.DoFn):

  """
  Wrapper class that allows subclasses of ObjectMapper to be used in Beam pipeline stages
    
  Attributes:
      mapper (ObjectMapper): Instance of an ObjectMapper's subclass
  
  """

  def __init__(self, mapper: ObjectMapper):

    if not isinstance(mapper, ObjectMapper):
      raise TypeError("mapper needs to be a subclass of ObjectMapper")
    self.mapper = mapper

  def process(self, data):
    return self.mapper.map(data)


# class ZipReader(ObjectMapper):
#   """
#   Loads the bytestream from a BytestreamPath object and returns an in memory zip holder object
  
#   """

#   def raw_map(self, path: BytestreamPath) -> t.Generator[InMemoryZipHolder, None, None]:
#     yield InMemoryZipHolder(filename = path.filename, content = path.read_bytes())



class InputToFontFiles(ObjectMapper):

  """
    Opens an in-memory zip holder and outputs individual font files

  """

  def raw_map(self, file: InMemoryZipHolder) -> t.Generator[InMemoryFontfileHolder,None,None]:

    def choose_ext(lst):
      ttfs = len([x for x in lst if ".ttf" in x.lower()])
      otfs = len([x for x in lst if ".otf" in x.lower()])
      if ttfs >= otfs:
        return ".ttf"
      else:
        return ".otf"

    #we assume the stream is a zip file's contents
    try:
      zipped = file.deserialise()
    except Exception as e:
      logger.exception(f"Error: source ({file.filename}) can't be read as zip")
      return
    files_in_zip = zipped.namelist()
    # choose whether to proces TTFs or OTFs, but not both
    ext = choose_ext(files_in_zip)
    valid_files = sorted([filename for filename in files_in_zip if ext in filename.lower()])
    
    for file in valid_files:
      filename = Path(file).name
      try: 
        content = zipped.read(file)
        yield InMemoryFontfileHolder(filename=filename, content = content)
      except Exception as e:
        logger.exception(f"Error while extracting file {filename} from zip")




class FontFileToLabeledChars(ObjectMapper):
  """
    Processes ttf files and outputs labeled examples consisting of a label (character), a numpy array corresponding to image features and a fontname string indicating the original filename

  """
  def __init__(
    self, 
    charset = string.ascii_letters + string.digits, 
    font_extraction_size = 100, 
    canvas_size = 500, 
    canvas_padding = 100):
    """
    
    Args:
        charset (str, optional): string containg all characters that are to be extracted from the font files
        font_extraction_size (int, optional): Font size to be used at extraction
        canvas_size (int, optional): Canvas array size in which to paste the extracted characters
        canvas_padding (int, optional): Padding to use when pasting the characters
    
    Raises:
        ValueError: Raised when the padding is too large for the provided canvas size
    """
    if canvas_padding >= canvas_size/2:
      raise ValueError(f"Canvas padding value ({canvas_padding}) is too large for canvas size ({canvas_size})")

    self.font_extraction_size = font_extraction_size
    self.canvas_size = canvas_size
    self.canvas_padding = canvas_padding
    self.canvas_size = canvas_size
    self.charset = charset

  def raw_map(self,file: InMemoryFontfileHolder)-> t.Generator[LabeledChar, None, None]:
    logger.info(f"exctracting arrays from file '{file.filename}'")
    try:
      font = file.deserialise(font_size = self.font_extraction_size)
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
        yield LabeledChar(features=array,label=char,fontname=file.filename)
      except Exception as e:
        logger.exception(f"Error while reading char '{char}' from font file '{file.filename}'")


class FeatureCropper(ObjectMapper):

  """
  Crops a labeled example's feature array  and returns the smallest bounding box containing all non-zero value.
  
  """

  def raw_map(self, example: LabeledChar) -> t.Generator[LabeledChar, None, None]:
    nonzero = np.where(example.features > 0)
    if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
      logger.info("Empty image found. ignoring.")
      return
      #yield key, LabeledChar(x=np.empty((0,),dtype=np.uint8), y=example.y)#(0, 0), (0,0)
    else:
      h_bound, w_bound = [(np.min(axis),np.max(axis)) for axis in nonzero]
      h = h_bound[1] - h_bound[0] + 1
      w = w_bound[1] - w_bound[0] + 1
      #crop and map to png
      cropped = example.features[h_bound[0]:(h_bound[0] + h),w_bound[0]:(w_bound[0]+w)]
      yield LabeledChar(features=cropped, label=example.label, fontname=example.fontname)


class FeatureResizer(ObjectMapper):

  """
    Resizes an image's numpy array to a square image with the specified dimensions

  """

  def __init__(self, output_size = 64):
    """
    
    Args:
        output_size (int, optional): height and width of output array
    """
    self.output_size = 64

  def raw_map(self, example: LabeledChar) -> t.Generator[LabeledChar, None, None]:
    """
    resize given image to a squared output image
    """
    array, y, metadata = example

    output = np.zeros((self.output_size,self.output_size),dtype=np.uint8)
    # resize img to fit into output dimensions
    try:
      height, width = example.features.shape
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
        yield LabeledChar(features=output.astype(np.uint8), label=y,fontname=metadata)
    except Exception as e:
      logger.exception(f"Error while resizing array: {e}")
      return




class FontFileToLabeledFont(FontFileToLabeledChars):
  """
    Processes ttf files and outputs a LabeledFont object consisting of labels and numpy arrays corresponding to image features for each character in the alphabet, and a fontname string indicating the original font filename

  """
  def raw_map(self,file: InMemoryFontfileHolder)-> t.Generator[LabeledFont, None, None]:
    imgs = []
    labels = []
    for mapped in super().raw_map(file):
      imgs.append(mapped.features)
      labels.append(mapped.label)

    yield LabeledFont(features=np.stack(imgs), label=np.array(labels), fontname = mapped.fontname) 



class FontMapper(ObjectMapper):

  """
  Applies an ObjectMapper transformation to every character in a LabeledFont object
  
  Attributes:
      mapper (ObjectMapper): Core transformation
  
  
  """
  def __init__(self, mapper: ObjectMapper):
    self.mapper = mapper

  def raw_map(self, alphabet: LabeledFont) -> t.Generator[LabeledFont, None, None]:
    imgs = []
    labels = []
    for example in alphabet:
      for mapped in self.mapper.raw_map(example):
        imgs.append(mapped.features)
        labels.append(mapped.label)

    yield LabeledFont(features=np.stack(imgs), label=np.array(labels), fontname = example.fontname)


class FeatureCropperAndResizer(ObjectMapper):
  
  """
  Crops and resizes character images in a single step; this is to be able to stack output images into a single numpy array in every pipeline stage for LabeledFont instances.
  
  Attributes:
      cropper (ObjectMapper)
      resizer (ObjectMapper)
  """
  def __init__(self, output_size = 64):
    """
    
    Args:
        output_size (int, optional): height and width of output array
    """

    self.cropper = FeatureCropper()
    self.resizer = FeatureResizer(output_size)

  def raw_map(self, example: LabeledChar) -> t.Generator[LabeledChar, None, None]:

    for cropped in self.cropper.raw_map(example):
      for resized in self.resizer.raw_map(cropped):
        yield resized




class Writer(beam.DoFn):
  """
  Takes instances of LabeledChar and writes them to a tensorflow record file.
  
  Attributes:
      output_path (str): Output path
  
  """
  def __init__(self, writer: BatchWriter):
    self.writer = writer
  def process(self,example: LabeledChar) -> None:
    try:
      self.writer.write(example)
    except Exception as e:
      logging.exception(f"error writing example {example}: {e}")

  def teardown(self):
    self.writer.close()



class PipelineFactory(object):

  """Factory class to construct core transformation sequence for preprocessing font files
  """
  
  @classmethod
  def create(cls, 
    output_record_class: type,
    charset: str,
    font_extraction_size: int,
    canvas_size: int,
    canvas_padding: int,
    output_array_size: int) -> PipelineExecutor:
    """Build file processing pipeline object
    
    Args:
        output_record_class (type): Class of output record schema, inheriting from TfrWritable
        charset (str): String with characters to be extracted
        font_extraction_size (int): Font size to use when extracting font images
        canvas_size (int): Image canvas size in which fonts will be extracted
        canvas_padding (int): Padding in the image extraction canvas
        output_array_size (int): Final character image size
    
    Returns:
        PipelineExecutor: Procesing transformation object
    """
    if output_record_class == LabeledChar:
      return PipelineExecutor(
        stages = [
        InputToFontFiles(),
        ManyToManyMapper(
          mapper = FontFileToLabeledChars(
            charset = charset,
            font_extraction_size = font_extraction_size,
            canvas_size = canvas_size,
            canvas_padding = canvas_padding)
        ),
        ManyToManyMapper(
          mapper = FeatureCropper()
        ),
        ManyToManyMapper(
          mapper = FeatureResizer(output_size = output_array_size)
        )]
      )

    elif output_record_class == LabeledFont:

      return PipelineExecutor(
        stages = [
        InputToFontFiles(),
        ManyToManyMapper(
          mapper = FontFileToLabeledFont(
            charset = charset,
            font_extraction_size = font_extraction_size,
            canvas_size = canvas_size,
            canvas_padding = canvas_padding)
        ),
        ManyToManyMapper(
          mapper = FontMapper(
            mapper = FeatureCropperAndResizer(output_size = output_array_size)
        ))]
      )

    else:

      raise TypeError(f"Output schema class not recognised: {output_record_class}")