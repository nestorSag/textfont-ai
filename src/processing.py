from PIL import Image, ImageFont, ImageDraw
import os
import imageio 
import string
import numpy as np
import zipfile
import re
from shutil import copyfile

from abc import ABC, abstractmethod


class ProcessingStage(ABC):

  def __init__(self):
    self.chars = string.ascii_letters + string.digits
    self.counter = 0

  @abstractmethod
  def process(self,**kwargs):
    pass

  def create_folder(self,folder: str):
    if not os.path.isdir(folder):
      os.makedirs(folder)

class ZipExtractor(ProcessingStage):

  def process(self,file,output_folder):
    self.create_folder(output_folder)
    try:
      with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    except zipfile.BadZipFile as e:
      print("skipped malformed zip")

class GoogleZipExtractor(ZipExtractor):

  def process(self,file,output_folder):
    super().process(file,"temp")
    self.create_folder(output_folder)
    self.collect_ttf_files("temp",output_folder)
    
  def collect_ttf_files(self,source,target):
    for obj in os.listdir(source):
      obj_path = source + "/" + obj
      #print(obj_path)
      if os.path.isdir(obj_path):
        self.collect_ttf_files(obj_path, target)
      else:
        if bool(re.search(r"Regular\.ttf$",obj_path)):
          obj_filename = obj_path.split("/")[-1]
          #print("source: {s}, filename: {f}, sink: {t}".format(s=obj_path, f=obj_filename,t=target + "/" + obj_filename))
          copyfile(obj_path,target + "/" + obj_filename)

class PngExtractor(ProcessingStage):

  def process(self,input_file,output_folder,size=128,font_size=100):
    self.create_folder(output_folder)
    for letter in self.chars:
      # output file name avoids overwriting
      filename = input_file.split("/")[-1]
      output_file = output_folder + "/" + letter + "@" + str(self.counter) + "@" + filename.replace("ttf","png")
      im = Image.new("RGB",(size*4,size*4))
      draw = ImageDraw.Draw(im)
      try:
        font = ImageFont.truetype(input_file,font_size)
        draw.text((int(size/2),int(size/2)),letter,font=font)
      except Exception as e:
        print("error: {e}".format(e=e))
      #im=im.crop(im.getbbox())
      im.resize((size,size)).save(output_file)
      self.counter += 1

class PngNormaliser(ProcessingStage):
  # finds a suitable image size to standardise all letters from all fonts, then maps original pngs to new size
  # centering each character
  def get_char_dim(self,gray_img):
    nonzero = np.where(gray_img > 0)
    if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
      return (0, 0), (0,0)
    else:
      dims = [(np.min(axis),np.max(axis)) for axis in nonzero]
      return dims

  def get_global_bounding_box(self,input_folder):

    max_height = 0
    max_width = 0
    for char in self.chars:
      for file in os.listdir(input_folder):
        filepath = input_folder + "/" + file
        gray_img = np.mean(imageio.imread(filepath),axis=-1)
        h_lim, w_lim = self.get_char_dim(gray_img)
        h = h_lim[1] - h_lim[0] + 1
        w = w_lim[1] - w_lim[0] + 1
        max_height = max(max_height,h)
        max_width = max(max_width,w)

    return max_height, max_width

  def process(self,input_file,output_folder,dim_tuple):
    make_folder(output_folder)
    H,W = dim_tuple
    #H,W = self.get_global_bounding_box(input_folder)
    #print("global bounding box: {h} x {w}".format(h=H,w=W))
    #for filename in os.listdir(input_folder):
    filename = input_file.split("/")[-1]
    gray_img = np.mean(imageio.imread(input_file),axis=-1)
    h_lim, w_lim = get_char_dim(gray_img)
    h = h_lim[1] - h_lim[0] + 1
    w = w_lim[1] - w_lim[0] + 1
    # paste char image roughly to canvas center
    h_padding, w_padding = int((H - h)/2), int((W - w)/2)
    canvas = np.zeros((H,W,1),dtype=np.uint8)
    canvas[h_padding:(h_padding+h),w_padding:(w_padding+w),0] = gray_img[h_lim[0]:(h_lim[1]+1),w_lim[0]:(w_lim[1]+1)]
    imageio.imwrite(output_folder + "/" + filename,canvas)



