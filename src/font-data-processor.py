from PIL import Image, ImageFont, ImageDraw
import os
import imageio 
import numpy as np

DATA_FOLDER = "data/fonts"
PNG_FOLDER = DATA_FOLDER + "/pngs"
#TTF_FOLDERS = [DATA_FOLDER + "/google/ttf", DATA_FOLDER + "/1001free/ttf"]
#CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

# test parameter values
TTF_FOLDERS = [DATA_FOLDER + "/google/ttf"]
CHARS = "A"

def make_folder(folder):
  if not os.path.isdir(folder):
    os.mkdir(folder)
## Python 3 code => has some issues 
def get_png_from_ttf(input_file,letter,output_file,size=250,font_size=100):
  im = Image.new("RGB",(size,size))
  draw = ImageDraw.Draw(im)
  font = ImageFont.truetype(input_file,font_size)
  draw.text((50,50),letter,font=font)
  #im=im.crop(im.getbbox())
  im.save(output_file)

def process_recursive(item, output_folder,size=100):
  if os.path.isfile(item):
    for char in CHARS:
      output_file = output_folder + "/" + char + "/" + item.split("/")[-1] + ".png"
      try:
        get_png_from_ttf(item,char,output_file,size=size)
      except Exception as e:
        print("error: {e}".format(e=e))
      #get_png_from_ttf(item,char,output_file,size=size)
  else:
    for subitem in os.listdir(item):
      process_recursive(item + "/" + subitem, output_folder)

def generate_pngs(output_size = 200):

  make_folder(PNG_FOLDER)

  for char in CHARS:
    letter_folder = PNG_FOLDER + "/" + char
    make_folder(letter_folder)

  for folder in TTF_FOLDERS:
    for item in os.listdir(folder):
      process_recursive(folder + "/" + item, PNG_FOLDER, output_size)

def get_char_dim(gray_img):
  nonzero = np.where(gray_img > 0)
  if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
    return (0, 0), (0,0)
  else:
    dims = [(np.min(axis),np.max(axis)) for axis in nonzero]
    return dims

def get_global_bounding_box():

  max_height = 0
  max_width = 0
  for char in CHARS:
    folder = PNG_FOLDER + "/" + char
    for file in os.listdir(folder):
      filepath = folder + "/" + file
      gray_img = np.mean(imageio.imread(filepath),axis=-1)
      h_lim, w_lim = get_char_dim(gray_img)
      h = h_lim[1] - h_lim[0] + 1
      w = w_lim[1] - w_lim[0] + 1
      max_height = max(max_height,h)
      max_width = max(max_width,w)

  return max_height, max_width

def format_pngs():
  output_folder = DATA_FOLDER + "/" + "processed_pngs"
  make_folder(output_folder)

  H,W = get_global_bounding_box()
  print("global bounding box: {h} x {w}".format(h=H,w=W))
  for char in CHARS:
    input_folder = PNG_FOLDER + "/" + char
    char_output_folder = output_folder + "/" + char
    make_folder(char_output_folder)
    for file in os.listdir(input_folder):
      filename = file.split("/")[-1]
      filepath = input_folder + "/" + file
      gray_img = np.mean(imageio.imread(filepath),axis=-1)
      h_lim, w_lim = get_char_dim(gray_img)
      h = h_lim[1] - h_lim[0] + 1
      w = w_lim[1] - w_lim[0] + 1
      # paste char image roughly to canvas center
      h_padding, w_padding = int((H - h)/2), int((W - w)/2)
      canvas = np.zeros((H,W,1),dtype=np.uint8)
      canvas[h_padding:(h_padding+h),w_padding:(w_padding+w),0] = gray_img[h_lim[0]:(h_lim[1]+1),w_lim[0]:(w_lim[1]+1)]
      imageio.imwrite(char_output_folder + "/" + filename,canvas)
if __name__ == "__main__":
  #generate_pngs()
  format_pngs()