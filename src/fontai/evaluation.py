import os
import numpy as np
import pandas as pd
from PIL import Image


class ValidationDataExaminer(object):
  def __init__(self,predicted_labels,true_labels,imgs,output_dir):
    self.imgs = imgs
    self.index_df = pd.DataFrame({"predicted":predicted_labels,"true":true_labels,"score":predicted_scores,"index":list(range(len(predicted_scores)))})
    self.output_dir = f"{output_dir}/misclassified"
  #
  def filter_model_results(self,predicted_label=None,true_label=None,ascending_order=True,n=32,misclassified_only=False):
    call_output_dir = f"{self.output_dir}/true_{true_label}_pred_{predicted_label}_{'asc' if ascending_order else 'desc'}"
    os.makedirs(call_output_dir)
    if predicted_label is None and true_label is None:
      filtered_df = self.index_df
    elif predicted_label is None and true_label is not None:
      filtered_df = self.index_df[self.index_df.true == true_label]
    elif true_label is None and predicted_label is not None:
      filtered_df = self.index_df[self.index_df.predicted == predicted_label]
    else:
      filtered_df = self.index_df[(self.index_df.predicted == predicted_label) & (self.index_df.true == true_label)]
      #
    if misclassified_only:
      filtered_df = filtered_df[filtered_df.true != filtered_df.predicted]
    filtered_df = filtered_df.sort_values(by="score",ascending = ascending_order)
    #
    counter = 0
    print(f"saving samples to {call_output_dir}")
    for i in range(n):
      row = filtered_df.iloc[i,:]
      pred = row["predicted"]
      true = row["true"]
      img = self.imgs[row["index"],:,:,0].reshape((64+2*padding,64+2*padding)).astype(np.uint8)
      im = Image.fromarray(img)
      im.save(f"{call_output_dir}/{counter}_score_{row['score']}_pred_{pred}_true_{true}.png")
      counter += 1