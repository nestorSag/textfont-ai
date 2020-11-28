from PIL import Image, ImageFont, ImageDraw
import os 

DATA_FOLDER = "data/fonts"
TTF_FOLDERS = [DATA_FOLDER + "/google/ttf", DATA_FOLDER + "/1001free/ttf"]
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"


## Python 3 code => has some issues 
def get_png_from_ttf(input_file,letter,output_file,size=250,font_size=100):
  im = Image.new("RGB",(size,size))
  draw = ImageDraw.Draw(im)
  font = ImageFont.truetype(input_file,font_size)
  draw.text((0,0),letter,font=font)
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

def process_ttfs(output_size = 200):

  png_folder = DATA_FOLDER + "/pngs"
  if not os.path.isdir(png_folder):
    os.mkdir(png_folder)

  for char in CHARS:
    letter_folder = png_folder + "/" + char
    if not os.path.isdir(letter_folder):
      os.mkdir(letter_folder) 

  for folder in TTF_FOLDERS:
    for item in os.listdir(folder):
      process_recursive(folder + "/" + item, png_folder, output_size)

if __name__ == "__main__":
  process_ttfs()