import numpy as np
import re
from PIL import Image

# access compressed numpy data
with np.load("test.npz") as data:
  filenames = data["filenames"]
  char = data["char"]
  img = data["img"]


#### find fonts with different styles
n = len(filenames)
styles = ["bold","italic","3d"]
styles_idx = {}
for style in styles:
  styles_idx[style] = [idx for idx in range(n) if re.search(style,filenames[idx])]

exclusive_styles_idx = {}
for style in styles:
  others = set()
  for other_style in styles:
    if other_style != style:
      others.update(tuple(styles_idx[other_style]))
  exclusive_styles_idx[style] = list(set(styles_idx[style]).difference(others))


#### find malformed images
x = img[0,:,:,:]
def is_malformed(img):
  m,n,k = img.shape
  img = img.reshape((m,n))
  a = np.sum(np.apply_along_axis(lambda x: np.any(x>0),arr=x,axis=0))
  b = np.sum(np.apply_along_axis(lambda x: np.any(x>0),arr=x,axis=1))
  return a <= 2 or b <= 2

z = np.apply_over_axes(lambda a,b: print(a),a=img,axes=(0))

N,m,n,k = img.shape
malformed = []
for i in range(N):
  if is_malformed(img[i,:,:,:]):
    malformed.append(i)

