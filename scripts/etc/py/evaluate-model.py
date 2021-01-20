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

from PIL import Image


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#parameters
val_data_dir = "./data/val/"
output_dir = "./models/model1_lowercase"
pixel_threshold = 0
padding=1
output_dir = "inspect/model1_lowercase"
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

# merge results in dataframe
imgs = np.empty((0,64+padding,64+padding,1))
for img, label in val_dataset:
  imgs = np.concatenate([imgs,(255*img.numpy()).astype(np.uint8)],axis=0)

class ValidationDataExaminer(object):
  def __init__(self,predicted_labels,true_labels,imgs,output_dir):
    self.imgs = imgs
    self.index_df = pd.DataFrame({"predicted":predicted_labels,"true":true_labels,"score":predicted_scores,"index":list(range(len(predicted_scores)))})
    self.output_dir = f"{output_dir}/misclassified"
  #
  def filter_model_errors(self,predicted_label=None,true_label=None,ascending_order=True,n=32):
    call_output_dir = f"{self.output_dir}/true_{true_label}_pred_{predicted_label}_{'asc' if ascending_order else 'desc'}"
    os.makedirs(call_output_dir)
    if predicted_label is None and true_label is None:
      filtered_df = self.index_df[self.index_df]
    elif predicted_label is None and true_label is not None:
      filtered_df = self.index_df[self.index_df.true == true_label]
    elif true_label is None and predicted_label is not None:
      filtered_df = self.index_df[self.index_df.predicted == predicted_label]
    else:
      filtered_df = self.index_df[(self.index_df.predicted == predicted_label) & (self.index_df.true == true_label)]
    filtered_df = filtered_df.sort_values(by="score",ascending = ascending_order)
    #
    counter = 0
    print(f"saving samples to {call_output_dir}")
    for i in range(n):
      row = filtered_df.iloc[i,:]
      pred = row["predicted"]
      true = row["true"]
      img = self.imgs[row["index"],:,:,0].reshape((64+padding,64+padding)).astype(np.uint8)
      im = Image.fromarray(img)
      im.save(f"{call_output_dir}/{counter}_score_{row['score']}_pred_{pred}_true_{true}.png")
      counter += 1

examiner = ValidationDataExaminer(predicted_labels,true_labels,imgs,output_dir)
examiner.filter_model_errors(true_label = "a",ascending_order=False)
examiner.filter_model_errors(true_label = "l",predicted_label="i",ascending_order=True)

# img_, label_ = next(iter(dataset))
# img = (255*img_.numpy()).astype(np.uint8)
# label = label_.numpy()

# for idx in range(128):
#   eximg = img[idx].reshape((67,67))
#   exlabel = handler.classes[np.argmax(label[idx,:])]
#   #
#   im = Image.fromarray(eximg)
#   im.save(f"batch/{idx}_{exlabel}.png")

# #Tensor("args_0:0", shape=(None, 67, 67, None), dtype=float32)
