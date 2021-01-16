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


handler = TFRHandler()
dataset = handler.get_tf_dataset(["./tmp/sample-14.tfr","./tmp/sample-15.tfr"],batch_size=32)

with open("tmp/hyperpars.json","r") as f:
	hyperpar_dict = json.loads(f.read())

#model = get_stacked_network(hyperpar_dict)

model = tf.keras.Sequential(
  [tf.keras.Input(shape = (64,64,1)),
   tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
   tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
   # tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
   # tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(62,activation="softmax")
  ]
)

model.compile(loss = hyperpar_dict["loss"], optimizer = hyperpar_dict["optimizer"], metrics = hyperpar_dict["metrics"])
model.summary()

# x_train, y_train, fontnames, chars = handler.files_to_numpy(["tmp/sample-15.tfr"])
# #labels = labels.reshape((-1,1))
# #x_train,  x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1)

# y_train = tf.keras.utils.to_categorical(y_train,num_classes)
# y_test = tf.keras.utils.to_categorical(y_test,num_classes)

# # y_train = tf.one_hot(y_train,num_classes)
# # y_test = tf.one_hot(y_test,num_classes)

# #train_mean = np.mean(x_train,axis=0)#
# #train_sd = np.std(x_train,axis=0)

# #x_train = ((x_train-train_mean)/train_sd).astype(np.float32)
# #x_test = ((x_test-train_mean)/train_sd).astype(np.float32)


# x_train = (x_train/255).astype(np.float32)
# x_test = (x_test/255).astype(np.float32)


#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)
#y_train = keras.utils.to_categorical(y_train,num_classes)
#y_test = keras.utils.to_categorical(y_test,num_classes)


#x_train_t, x_test_t, y_train_t, y_test_t = tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#/etc/modprobe.d/nvidia-kernel-common.conf

model.fit(dataset, epochs = 25,callbacks=[tensorboard_callback])

# test_predicted = model.predict(x_test)

# truth_test = [handler.classes[x] for x in np.argmax(y_test,1)]
# predicted_test = [handler.classes[x] for x in np.argmax(test_predicted,1)]

# cm = confusion_matrix(truth_test, predicted_test)
# # Log the confusion matrix as an image summary.
# figure = plot_confusion_matrix(cm, class_names=list(handler.classes))
# cm_image = plot_to_image(figure)







# def plot_confusion_matrix(cm, class_names):
#   """
#   Returns a matplotlib figure containing the plotted confusion matrix.

#   Args:
#     cm (array, shape = [n, n]): a confusion matrix of integer classes
#     class_names (array, shape = [n]): String names of the integer classes
#   """
#   figure = plt.figure(figsize=(8, 8))
#   plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#   plt.title("Confusion matrix")
#   plt.colorbar()
#   tick_marks = np.arange(len(class_names))
#   plt.xticks(tick_marks, class_names, rotation=45)
#   plt.yticks(tick_marks, class_names)
#   #
#   # Compute the labels from the normalized confusion matrix.
#   labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
#   #
#   # Use white text if squares are dark; otherwise black.
#   threshold = cm.max() / 2.
#   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     color = "white" if cm[i, j] > threshold else "black"
#     plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
#     #
#   plt.tight_layout()
#   plt.ylabel('True label')
#   plt.xlabel('Predicted label')
#   return figure

import string
import json 
import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fontai.preprocessing import TFRHandler
from fontai.models import * 
import tensorflow as tf

classes = tf.convert_to_tensor(list(string.ascii_letters + string.digits))
num_classes = 62

dataset = tf.data.TFRecordDataset(filenames=["./tmp/sample-14.tfr","./tmp/sample-15.tfr"])

serialized = next(iter(dataset))

record_spec = {
  'char': tf.io.FixedLenFeature([], tf.string),
  'filename': tf.io.FixedLenFeature([], tf.string),
  'img': tf.io.FixedLenFeature([], tf.string),
}

def parse_record(serialized):
  parsed = tf.io.parse_single_example(serialized,record_spec)
  img = (tf.image.decode_png(parsed["img"])/255)
  label = np.zeros((len(classes),),dtype=np.float32)
  #print(parsed["char"])
  #print(type(parsed["char"]))
  label = tf.keras.utils.to_categorical(tf.where(classes == parsed["char"]),num_classes=num_classes).reshape((num_classes,))
  return img, label

dataset2 = dataset.map(parse_record)

