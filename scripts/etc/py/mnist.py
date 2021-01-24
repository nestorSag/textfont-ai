import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

num_classes = 10
input_shape = (28,28,1)

(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)


### sequential model
model = keras.Sequential(
  [keras.Input(shape = input_shape),
   layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
   layers.MaxPooling2D(pool_size=(2,2)),
   layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
   layers.MaxPooling2D(pool_size=(2,2)),
   layers.Flatten(),
   layers.Dropout(0.5),
   layers.Dense(num_classes,activation="softmax")
  ]
)

model.summary()


batch_size = 128
epochs = 1

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(x_train,y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)

model.save("model")
model = tf.keras.models.load_model("model")

score = model.evaluate(x_test,y_test,verbose=0)
