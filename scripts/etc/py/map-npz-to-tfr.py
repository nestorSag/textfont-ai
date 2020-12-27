from google.cloud import storage
import io
import imageio
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

storage_client = storage.Client()
bkt ="textfont-ai-data"
bucket = storage_client.bucket(bkt)

folders = ["processed/npy/64/sorted-by-hash","processed/npy/64/sorted-by-char"]
prf_dict = {}
for folder in folders:
  preffixes = []
  for file in storage_client.list_blobs('textfont-ai-data', prefix=folder):
    preffixes.append(file.public_url.split("/")[-1].split("-")[0])
  preffixes = list(set(preffixes))
  prf_dict[folder] = preffixes

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def img_to_png_bytes(img):
  bf = io.BytesIO()
  imageio.imwrite(bf,img,"png")
  val = bf.getvalue()
  bf.close()
  return val

for folder in prf_dict:
  for preffix in prf_dict[folder]:
    objs = {}
    dst_path = (folder + "/" + preffix).replace("npy","tf") + ".tfr"
    if not storage.Blob(bucket=bucket,name=dst_path).exists():
      for kind in ["img","char","filenames"]:
        src_path = folder + "/" + preffix + "-{k}.npy".format(k=kind)
        print("downloading {x}".format(x=src_path))
        objs[kind] = np.load(io.BytesIO(bucket.blob(src_path).download_as_bytes()))
      print("uploading {x}".format(x=dst_path))
      with tf.io.TFRecordWriter("gs://" + bkt + "/" + dst_path) as writer:
        n,h,w,c = objs["img"].shape
        for i in range(n):
          img = img_to_png_bytes(objs["img"][i,:,:,0].reshape((h,w)))
          char = str.encode(str(objs["char"][i]))
          filename = str.encode(str(objs["filenames"][i]),errors="replace")
          example = tf.train.Example(
            features=tf.train.Features(
              feature={
              "img": _bytes_feature(img),
              "char":_bytes_feature(bytes(char)),
              "filename":_bytes_feature(bytes(filename))}))
          writer.write(example.SerializeToString())
