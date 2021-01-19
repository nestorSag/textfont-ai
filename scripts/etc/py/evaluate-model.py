import os
import json 
import datetime
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from fontai.preprocessing import TFRHandler
from fontai.models import * 
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#parameters
val_data_dir = "./data/val/"
model_path = "models/model1/model"
pixel_threshold = 0
padding=3
output_dir = "inspect/model2"
charset = "lowercase"

os.makedirs(output_dir)

# instantiate objects
handler = TFRHandler(padding=padding,pixel_threshold=pixel_threshold,charset=charset)
model = tf.keras.models.load_model(model_path)
#### get validation metrics
val_dataset = handler.get_evaluation_dataset(folder=val_data_dir)
eval_metric = dict(zip(["loss","accuracy"],model.evaluate(val_dataset)))

#### analyse validation results
predicted_raw = model.predict(val_dataset)
predicted_scores = np.max(predicted_raw,axis=1)
predicted_labels = np.array([handler.classes[x] for x in np.argmax(predicted_raw,axis=1)])

true_labels = np.concatenate([[handler.classes[z] for z in np.argmax(y,axis=1)] for x,y in val_dataset])


# get confusion matrix
predmat = confusion_matrix(true_labels,predicted_labels,normalize="true")
predmat = pd.DataFrame(predmat)
predmat.columns = list(handler.classes)
predmat.index = list(handler.classes)

plt.rcParams.update({'font.size': 8})
plt.figure(figsize = (20,20))
sn.heatmap(predmat, annot=True)
plt.savefig(output_dir + "/" + "heatmap.png")
#predmat.to_csv(output_dir + "/" + "confusion matrix.csv")


val_dataset = handler.
from PIL import Image
import shutil

img_, label_ = next(iter(dataset))
img = (255*img_.numpy()).astype(np.uint8)
label = label_.numpy()

shutil.rmtree("batch")
os.mkdir("batch")
for idx in range(128):
  eximg = img[idx].reshape((67,67))
  exlabel = handler.classes[np.argmax(label[idx,:])]
  #
  im = Image.fromarray(eximg)
  im.save(f"batch/{idx}_{exlabel}.png")

#Tensor("args_0:0", shape=(None, 67, 67, None), dtype=float32)
