import tensorflow as tf

record_spec = {
    'char': tf.io.FixedLenFeature([], tf.string),
    'filename': tf.io.FixedLenFeature([], tf.string),
    'img': tf.io.FixedLenFeature([], tf.string),
}

filenames = ["sample.tfr"]
records = tf.data.TFRecordDataset(filenames)

def parse_record(serialized):
  return tf.io.parse_single_example(serialized,img_spec)

examples = dt.map(parse_record)

objs = []
counter = 0
for example in examples:
    objs.append(example["img"])
