import json 
import numpy as np

from sklearn.model_selection import train_test_split
from fontai.preprocessing import TFRHandler
from fontai.models import * 

num_classes=62

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

with open("tmp/hyperpars.json","r") as f:
	hyperpar_dict = json.loads(f.read())

#model = get_stacked_network(hyperpar_dict)

model = tf.keras.Sequential(
  [tf.keras.Input(shape = (64,64,1)),
   tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
   tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
   tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(62,activation="softmax")
  ]
)
model.compile(loss = hyperpar_dict["loss"], optimizer = hyperpar_dict["optimizer"], metrics = hyperpar_dict["metrics"])
model.summary()

handler = TFRHandler()
imgs, labels, fontnames, chars = handler.to_numpy("tmp/sample.tfr")
#labels = labels.reshape((-1,1))
x_train,  x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1)

y_train = tf.keras.utils.to_categorical(y_train,num_classes)
y_test = tf.keras.utils.to_categorical(y_test,num_classes)

# y_train = tf.one_hot(y_train,num_classes)
# y_test = tf.one_hot(y_test,num_classes)


x_train = (x_train/255).astype(np.float32)
x_test = (x_test/255).astype(np.float32)
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)
#y_train = keras.utils.to_categorical(y_train,num_classes)
#y_test = keras.utils.to_categorical(y_test,num_classes)


#x_train_t, x_test_t, y_train_t, y_test_t = tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)


model.fit(x_train,y_train, batch_size = 128, epochs = 100, validation_split = 0.1)
