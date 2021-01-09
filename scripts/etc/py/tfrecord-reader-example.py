import tensorflow as tf

class TFRHandler(object):

  def __init__(self):

    self.record_spec = {
        'char': tf.io.FixedLenFeature([], tf.string),
        'filename': tf.io.FixedLenFeature([], tf.string),
        'img': tf.io.FixedLenFeature([], tf.string),
    }

  def parse_record(serialized):
    return tf.io.parse_single_example(serialized,self.record_spec)


  def to_numpy(self,filepath):
    records = tf.data.TFRecordDataset(filepath)
    examples = records.map(parse_records)

    mgs = []
    filenames = []
    chars = []
    for example in examples:
      with io.BytesIO() as bf:
      img = imageio.imread(io.BytesIO(example["img"].numpy()))
      imgs.append(img.reshape((1,) + img.shape + (1,)))
      filenames.append(example["filename"].numpy().decode("utf-8"))
      chars.append(example["char"].numpy().decode("utf-8"))

    imgs = np.concatenate(imgs,axis=0)
    return imgs, np.array(filenames) np.array(chars)
