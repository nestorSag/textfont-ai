from google.cloud import storage
import io
import numpy as np

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

for folder in prf_dict:
  for preffix in prf_dict[folder]:
    objs = {}
    dst_path = (folder + "/" + preffix).replace("npy","npz") + ".npz"
    if not storage.Blob(bucket=bucket,name=dst_path).exists():
      for kind in ["img","char","filenames"]:
        src_path = folder + "/" + preffix + "-{k}.npy".format(k=kind)
        print("downloading {x}".format(x=src_path))
        objs[kind] = np.load(io.BytesIO(bucket.blob(src_path).download_as_bytes()))
      print("uploading {x}".format(x=dst_path))
      dst_bf = io.BytesIO()
      np.savez_compressed(dst_bf,img=objs["img"],char=objs["char"],filenames=objs["filenames"])
      dst = bucket.blob(dst_path).upload_from_string(dst_bf.getvalue())
      dst_bf.close()
