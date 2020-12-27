import tensorflow as tf

record_spec = {
    'char': tf.io.FixedLenFeature([], tf.string),
    'filename': tf.io.FixedLenFeature([], tf.string),
    'img': tf.io.FixedLenFeature([], tf.string),
}

filenames = ["sample.tfr"]
records = tf.data.TFRecordDataset(filenames)

def parse_record(serialized):
  return tf.io.parse_single_example(serialized,record_spec)

examples = records.map(parse_record)

imgs = []
filenames = []
chars = []
for example in examples:
  with io.BytesIO() as bf:
    img = imageio.imread(io.BytesIO(example["img"].numpy()))
  imgs.append(img.reshape((1,) + img.shape + (1,)))
  filenames.append(example["filename"].numpy().decode("utf-8"))
  chars.append(example["char"].numpy().decode("utf-8"))

ims = np.concatenate(imgs,axis=0)
