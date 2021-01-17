import os
import json 
import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fontai.preprocessing import TFRHandler
from fontai.models import * 
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

num_classes=62
batch_size = 128
training_data_dir = "./data/train/"

handler = TFRHandler()
dataset = handler.get_tf_dataset([training_data_dir + file for file in os.listdir(training_data_dir)],batch_size=batch_size)

with open("tmp/hyperpars.json","r") as f:
	hyperpar_dict = json.loads(f.read())

#model = get_stacked_network(hyperpar_dict)

model = tf.keras.Sequential(
  [tf.keras.Input(shape = (64,64,1)),
   tf.keras.layers.Conv2D(32,kernel_size=(5,5),activation="relu"),
   tf.keras.layers.Conv2D(64,kernel_size=(4,4),activation="relu",strides=4),
   tf.keras.layers.Conv2D(96,kernel_size=(4,4),activation="relu"),
   # tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
   tf.keras.layers.Conv2D(128,kernel_size=(3,3),activation="relu",strides=3),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dropout(0.5),
   #tf.keras.layers.Dense(100,activation="relu"),
   tf.keras.layers.Dense(62,activation="softmax")
  ]
)


model.compile(loss = hyperpar_dict["loss"], optimizer = hyperpar_dict["optimizer"], metrics = hyperpar_dict["metrics"])
model.summary()


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#/etc/modprobe.d/nvidia-kernel-common.conf

model.fit(dataset, steps_per_epoch = int(8000000/batch_size), epochs = 1, callbacks=[tensorboard_callback])


# import string
# import json 
# import datetime
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from fontai.preprocessing import TFRHandler
# from fontai.models import * 
# import tensorflow as tf

# classes = tf.convert_to_tensor(list(string.ascii_letters + string.digits))
# num_classes = 62

# dataset = tf.data.TFRecordDataset(filenames=["./tmp/sample-14.tfr","./tmp/sample-15.tfr"])

# serialized = next(iter(dataset))

# record_spec = {
#   'char': tf.io.FixedLenFeature([], tf.string),
#   'filename': tf.io.FixedLenFeature([], tf.string),
#   'img': tf.io.FixedLenFeature([], tf.string),
# }

# def parse_record(serialized):
#   parsed = tf.io.parse_single_example(serialized,record_spec)
#   img = (tf.image.decode_png(parsed["img"])/255)
#   label = np.zeros((len(classes),),dtype=np.float32)
#   #print(parsed["char"])
#   #print(type(parsed["char"]))
#   label = tf.keras.utils.to_categorical(tf.where(classes == parsed["char"]),num_classes=num_classes).reshape((num_classes,))
#   return img, label

# dataset2 = dataset.map(parse_record)

