from __future__ import absolute_import
from collections.abc import Iterable
import os
import logging
import string
import zipfile
import io
from datetime import datetime
import typing as t

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import imageio
import tensorflow as tf

import apache_beam as beam
from apache_beam.io.gcp.gcsio import GcsIO


class ObjectMapper(ABC):
  """
    Interface for pre-ML file and data transformations

  """

  @abstractmethod
  def _map(self,data):
    pass

  def map(self,data):

    """
    Processes a single data instance.

    Returns a generator with a variable number of derived data instances

    """
    output = self._map(data)
    if not isinstance(output, t.GeneratorType):
      raise TypeError("Output of transform() must be a generator")
    return output

class LocalFileLoader(ObjectMapper):
  """
    Loads the bytestream from a local zip file

  """

  def _map(self, file: Path) -> InMemoryFile:
    yield InMemoryFile(name = file.name, content = file.read_bytes())

class GCSFileLoader(ObjectMapper):
  """
    Loads the bytestream from a zip file stored in GCP

  """

  def _map(self, path: str):
    yield zipfile.ZipFile(io.BytesIO(GcsIO().open(path,mode="r").read()))

class ZipToFontFiles(ObjectMapper):

  """
    Opens an in-memory zip file and outputs individual ttf files

  """

  def _map(self,file: InMemoryFile)-> t.Generator[InMemoryFile]:
    with io.BytesIO(file.content) as bf:
      zipped = zipfile.ZipFile(bf)
      for zipped_file in zipped.namelist():
        yield InMemoryFile(filename = zipped_file, content = zipped.read(zipped_file))

class FontFileToCharArrays(ObjectMapper):
  """
    Processes ttf files and outputs numpy arrays corresponding to individual character images

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

  def _map(self,file: InMemoryFile)-> t.Generator[np.ndarray]:
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
        yield array

class ArrayCropper(ObjectMapper):

  """
    Crops an array and returns an array corresponding to the bounding box containing all non-zero value.

  """

  def _map(self, array: np.ndarray) -> np.ndarray:

    nonzero = np.where(img > 0)
    if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
      yield np.empty((0,),dtype=np.uint8) #(0, 0), (0,0)
    else:
      h_bound, w_bound = [(np.min(axis),np.max(axis)) for axis in nonzero]
      h = h_bound[1] - h_bound[0] + 1
      w = w_bound[1] - w_bound[0] + 1
      #crop and map to png
      cropped = array[h_bound[0]:(h_bound[0] + h),w_bound[0]:(w_bound[0]+w)]
      yield cropped

class ArrayResizer(ObjectMapper):

  """
    Resizes an image's numpy array to a square image with the specified dimensions

    output_size: height and width of output array

  """

  def __init__(self, output_size = 64):
    self.output_size = 64

  def map(self, array):
    """
    resize given image to a squared output image
    """
    output = np.zeros((self.output_size,self.output_size),dtype=np.uint8)
    # resize img to fit into output dimensions
    img_h, img_w = array.shape
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
      yield output.astype(np.uint8)


class BeamCompatibleWrapper(beam.DoFn):

  """
    Wrapper that allows subclasses of ObjectWrapper to be used in Beam pipelines

    mapper: Instance of an ObjectWrapper's subclass

  """

  def __init__(self, mapper: ObjectMapper):

    if not isinstance(obj, ObjectMapper):
      raise TypeError("mapper needs to be a subclass of ObjectMapper")
    self.mapper = mapper

  def process(self, data):
    return self.mapper.map(data)


    

class BeamPipelineRunner(object):

  def __init__(self, config: PreprocessingConfig):
    self.config = config

  def run():

    with beam.pipelines(options=config.pipeline_options) as p:
      
      if user_options.input_folder[0:5] == "gs://":
        input_files = GcsIO().list_prefix(user_options.input_folder)
        input_files_list = list(input_files.keys())
      else:
        input_files = os.listdir(user_options.input_folder)
      #print("input_file_list: {l}".format(l=input_files_list))

      files = p | beam.Create(input_files_list)# ReadFromText(input_file_list_path)

      # unzip files, extract character PNGs from TTFs, crop PNGs to minimal size
      standardised = (files 
      | 'getPNGs' >> beam.ParDo(ImageExtractor(
          font_size=int(user_options.font_size),
          canvas_size=int(user_options.canvas_size),
          offset=int(user_options.png_offset)))
      | 'cropAndGroup' >> beam.ParDo(Cropper())
      | "standardisePNGs" >> beam.ParDo(Resizer(output_dim=int(user_options.png_size))))

      output_folder = user_options.output_folder if user_options.output_folder[-1] == "/" else user_options.output_folder + "/"
      if output_folder[0:5] != "gs://":
        Path(output_folder).mkdir(parents=True, exist_ok=True)

      sorted_by_hash = (standardised
        | 'setHashAsKey' >> beam.Map(lambda x: set_hash_as_key(x))
        | 'groupByHash' >> beam.GroupByKey()
        | 'createHashBundles' >> beam.ParDo(TFRecordContentCreator(int(user_options.png_size)))
        | 'saveHashToTFR' >> beam.ParDo(TFRecordUploader(output_folder + "sorted-by-hash")))
# class PreprocessingPipeline(object):

#   def __init__(self, stages = t.List[t.Tuple[str,ObjectMapper]]):
#     self.stages = stages

#   def fit(self,data):
#     self.call_stage_method(data,"fit")

#   def map(self,data):
#     self.call_stage_method(data,"map")

#   def call_method(self,data,method_name):
#     p_input = data
#     for stage in self.stages:
#       output = getattr(stage,method_name)(output)
#     return output


# class Preprocessor(object):

#   @staticmethod
#   def get_bounding_box(gray_img):
#     """
#     Takes an image and returns coordinates for smallest bounding box for nonzero elements

#     Returns:
#     (h_low,h_high), (w_low,w_high)
#     """
#     nonzero = np.where(gray_img > 0)
#     if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
#       return (0, 0), (0,0)
#     else:
#       dims = [(np.min(axis),np.max(axis)) for axis in nonzero]
#       return dims

#   @staticmethod
#   def get_filename(str,ext=".ttf"):
#     """
#     Extract filename from file path, assumed to be font name
#     """
#     return str.split("/")[-1].lower().replace(ext,"")

#   @staticmethod
#   def choose_ext(lst):
#     """
#     choose an extension to work on based on the extension's frequency in a list of zipped files.
#     """
#     ttfs = len([x for x in lst if ".ttf" in x.lower()])
#     otfs = len([x for x in lst if ".otf" in x.lower()])
#     if ttfs >= otfs:
#       return ".ttf"
#     else:
#       return ".otf"

#   @staticmethod
#   def extract_imgs_from_zip(zip_,canvas_size,font_size,offset):
#     """
#     Generator that extracts multiple images as numpy arrays (one per character) from a zip file of font files

#     Returns:
#     triplet with character, filename and numpy array
#     """
#     files_in_zip = zip_.namelist()
#     # choose whether to proces TTFs or OTFs, but not both
#     ext = Preprocessor.choose_ext(files_in_zip)
#     available = sorted([filename for filename in files_in_zip if ext in filename.lower()])
#     valid_files = []
#     #identifiers = []
#     for filename in available:
#       fontname = Preprocessor.get_filename(filename)
#       try:
#         valid_files.append((filename,zip_.read(filename)))
#       except Exception as e:
#         logging.exception("Error reading font file {x}: {e}".format(x=filename,e=e))
#     for name, file in valid_files:
#       for letter in string.ascii_letters + string.digits:
#         #logging.info("working on letter {l}".format(l=letter))
#         try:
#           letter_bf = io.BytesIO(file)
#           im = Image.new("RGB",(canvas_size,canvas_size))
#           draw = ImageDraw.Draw(im)
#           font = ImageFont.truetype(letter_bf,font_size)
#           #font = ImageFont.truetype(bytes(ttf),self.font_size)
#           draw.text((offset,offset),letter,font=font)
#           # filename indexes letter and font type, avoids overwritting data with timestamp
#           output_filename = Preprocessor.get_filename(name).lower()# + str(datetime.now().time())
#           letter_bf.close()
#           yield letter, output_filename, im
#         except Exception as e:
#           logging.exception("error processing letter {x} from file {l}: {e}".format(l=file,x=letter,e=e))
#           return 

#   @staticmethod
#   def crop_img(im):
#     """
#     Crops numpy array according to smallest bounding box for non-zero elements
#     """
#     bf = io.BytesIO()
#     im.save(bf,format="png")
#     # get bounding box dimension
#     gray_img = np.mean(imageio.imread(bf.getvalue(),format="png"),axis=-1).astype(np.uint8)
#     bf.close()
#     h_bound, w_bound = Preprocessor.get_bounding_box(gray_img)
#     if h_bound == (0,0) or w_bound == (0,0):
#       return
#     else:
#       h = h_bound[1] - h_bound[0] + 1
#       w = w_bound[1] - w_bound[0] + 1
#       #crop and map to png
#       cropped = gray_img[h_bound[0]:(h_bound[0] + h),w_bound[0]:(w_bound[0]+w)]
#       return cropped

#   @staticmethod
#   def resize(img,output_dim):
#     """
#     resize given image to a squared output image
#     """
#     output = np.zeros((output_dim,output_dim),dtype=np.uint8)
#     # resize img to fit into output dimensions
#     img_h, img_w = img.shape
#     if img_h > 0 and img_w > 0:
#       if img_h >= img_w:
#         resize_dim = (output_dim,int(img_w*output_dim/img_h))
#       else:
#         resize_dim = (int(img_h*output_dim/img_w),output_dim)
#       #try:
#       resized = np.array(Image.fromarray(np.uint8(img)).resize(size=tuple(reversed(resize_dim))))
#       # embed into squared image
#       resized_h, resized_w = resized.shape
#       h_pad, w_pad = int((output_dim - resized_h)/2), int((output_dim - resized_w)/2)
#       output[h_pad:(h_pad+resized_h),w_pad:(w_pad+resized_w)] = resized
#       # make the image binary
#       #output[output > int(255/2)] = 255
#       #output[output < int(255/2)] = 0
#       return output.astype(np.uint8)
#       # except Exception as e:
#       #   logging.exception("error resizing png: {e}".format(e=e))
#       #   return 



class InputDataHandler(object):

  def __init__(self,padding=2,pixel_threshold=100,charset="all",img_dim=(64,64)):

    self.img_dim = img_dim

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
    self.charset = charset

  def parse_tf_objects(self,serialized):
    return tf.io.parse_single_example(serialized,self.record_spec)

  def filter_by_char(self,parsed):
    return tf.reduce_any(self.tf_classes == parsed["char"])

  def process_tf_objects(self,parsed):
    img = tf.image.decode_png(parsed["img"])
    img = tf.image.resize_with_crop_or_pad(img,target_height=self.img_dim[0]+2*self.padding,target_width=self.img_dim[1]+2*self.padding)
    img = tf.cast(img,dtype=tf.float32)
    y = tf.cast(tf.where(self.tf_classes == parsed["char"]),dtype=tf.int32)
    label = tf.reshape(tf.one_hot(indices=y,depth=self.num_classes),(self.num_classes,))#.reshape((num_classes,))
    return img, label, parsed["filename"]

  # def filter_sparse_images(self,img,label):
  #   return tf.math.count_nonzero(img) > self.pixel_threshold

  def filter_sparse_images(self,img,label,filename):
    return tf.math.count_nonzero(img) > self.pixel_threshold

  @classmethod
  def remove_filename(self,img,label,filename):
    return img, label

  def get_dataset(self,folders,include_fontname=False):

    if not isinstance(folders,list):
      folders = [folders]

    def standardise_folder_name(folder):
      if folder[-1] != "/":
        folder = folder + "/"

      return folder

    folders = [standardise_folder_name(folder) for folder in folders]

    files = [folder + file for folder in folders for file in os.listdir(folder)]

    dataset = tf.data.TFRecordDataset(filenames=files)\
      .map(self.parse_tf_objects)\
      .filter(self.filter_by_char)\
      .map(self.process_tf_objects)\
      .filter(self.filter_sparse_images)

    return dataset

  def scramble_dataset(self,dataset,batch_size=32):
    dataset = dataset\
      .shuffle(buffer_size=2*batch_size)\
      .repeat()\
      .batch(batch_size)

    return dataset

  def get_training_dataset(self,folders,batch_size=32):
    return self.scramble_dataset(self.get_dataset(folders).map(InputDataHandler.remove_filename),batch_size=batch_size)

  def get_evaluation_dataset(self,folders,batch_size=None):
    dataset = self.get_dataset(folders).map(InputDataHandler.remove_filename)
    
    if batch_size is not None:
      return dataset.batch(batch_size)
    else:
      return dataset

  def supervised_filter(self,model):
    def filter_func(imgs,labels,filenames):
      # filters a batch using a trained model
      pred = model(imgs)
      condition = tf.argmax(pred,axis=-1) == tf.argmax(labels,axis=-1)
      return imgs[condition], labels[condition], filenames[condition]

    return filter_func

  def expand_labels(self,img,label,filename):

    lowercase_labels = tf.zeros((26,)) if self.charset != "lowercase" else label
    uppercase_labels = tf.zeros((26,)) if self.charset != "uppercase" else label
    number_labels = tf.zeros((10,)) if self.charset != "numbers" else label

    return img, tf.concat([lowercase_labels,uppercase_labels,number_labels],axis=0), filename

  @classmethod
  def read_gen_model_data(self,path,padding=0,img_dim=(64,64)):
    elemspec= (tf.TensorSpec(shape=(img_dim[0]+2*padding,img_dim[1]+2*padding,1), dtype=tf.float32, name=None), tf.TensorSpec(shape=(62,), dtype=tf.float32, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None))

    return tf.data.experimental.load(path,element_spec = elemspec,compression="GZIP")

  @classmethod
  def filter_by_char(cls,char):
    idx = list(string.ascii_letters + string.digits).index(char)
    if char == "all":
      def f():
        return True
    else:
      def f(img,label,*args):
        return tf.argmax(label) == idx

    return f
