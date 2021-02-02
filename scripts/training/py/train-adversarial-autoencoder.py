import os
import argparse
import json 
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from fontai.preprocessing import InputDataHandler
from fontai.evaluation import ValidationDataExaminer

from fontai.models import * 
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# training procedure
def fit_model(argv=None):
  parser = argparse.ArgumentParser(description = "Trains an adversarial autoencoder model using on character image data. Models are checkpointed every epoch.")
  parser.add_argument(
      '--input-dataset',
      dest='input_data',
      required = True,
      action="append",
      help='File of folder of Tensorflow Dataset input; thisa rgument can be repeated for multiple foles or folders.')
  parser.add_argument(
      '--output-folder',
      dest='output_folder',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      required = True,      
      help='Output folder where model will be saved')
  parser.add_argument(
      '--batch-size',
      dest='batch_size',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default=64,
      type=int,
      help='Batch size for trianing')
  parser.add_argument(
      '--epochs',
      dest='n_epochs',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default=3,
      type=int,
      help='Number of epochs for training')
  parser.add_argument(
      '--steps-per-epoch',
      dest='steps_per_epoch',
      type=int,
      default=50000,
      help='Steps per epoch for training.')
  parser.add_argument(
      '--encoder-hyperparameter-file',
      dest='encoder_hyperparameters',
      required=True,
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      help='JSON file with hyperparameter specifications for the encoding model.')
  parser.add_argument(
      '--decoder-hyperparameter-file',
      dest='decoder_hyperparameters',
      required=True,
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      help='JSON file with hyperparameter specifications for the decoding model.')
  parser.add_argument(
      '--discr-hyperparameter-file',
      dest='discr_hyperparameters',
      required=True,
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      help='JSON file with hyperparameter specifications for the discriminator model.')
  parser.add_argument(
      '--reuse-model',
      action="store_true",
      dest="reuse",
      help='Load model in output directory and use it for training.')
  parser.add_argument(
      '--lr-shrink-factor',
      default=1.0,
      type=float,
      dest="shrink_factor",
      help='shrink leearning rate by this every epoch; defaults to 1.0 (constant learning rate).')
  parser.add_argument(
      '--optimizer-dict',
      default=None,
      type=str,
      dest="optim_dict",
      help="JSON file with the model's optimizer specification; if not given, the default Adam optimizer is used")
  parser.add_argument(
      '--code-size',
      default=32,
      type=int,
      dest="code_size",
      help="Size of encoder output and discriminator input")

  args, _ = parser.parse_known_args(argv)

  output_dir = args.output_folder
  if output_dir[-1] != "/":
    output_dir += "/"

  #Load dataset
  dataset = tf.data.Dataset.from_tensor_slices(args.input_data)\
    .interleave(map_func=InputDataHandler.read_gen_model_data,cycle_length=8,block_length=16)\
    .map(InputDataHandler.remove_filename)\
    .prefetch(1000)\
    .shuffle(buffer_size=2*args.batch_size)\
    .repeat()\
    .batch(args.batch_size)

  #build model
  if args.reuse:
    print("reusing saved model in output directory..")
    model = SupervisedAdversarialAutoEncoder.load(output_dir)
  else:

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_hyperpars = {}

    #Initialise encoder
    with open(args.encoder_hyperparameters,"r") as f:
      hyperparameters = json.loads(f.read())

    all_hyperpars["encoder"] = hyperparameters

    output_size = args.code_size
    hyperparameters["layers"] = [{
      "class":"tf.keras.Input",
      "kwargs": {"shape":[64,64,1]}
    }] + hyperparameters["layers"] + [{
      "class":"tf.keras.layers.Dense",
      "kwargs": {"units":output_size}
    }]

    encoder = get_stacked_network(hyperparameters)

    # Initialise decoder
    with open(args.decoder_hyperparameters,"r") as f:
      hyperparameters = json.loads(f.read())

    all_hyperpars["decoder"] = hyperparameters

    input_size = args.code_size
    hyperparameters["layers"] = [{
      "class":"tf.keras.Input",
      "kwargs": {"shape":input_size + 62}
    }]

    hyperparameters["layers"][-1]["activation"] = "sigmoid"

    decoder = get_stacked_network(hyperparameters)
    decoder.add(tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0))
    
    #Initialise discriminator
    with open(args.discr_hyperparameters,"r") as f:
      hyperparameters = json.loads(f.read())

    all_hyperpars["discr"] = hyperparameters

    #output_size = len(handler.classes)
    hyperparameters["layers"] = [{
      "class":"tf.keras.Input",
      "kwargs": {"shape":args.code_size}
    }] + hyperparameters["layers"] + [{
      "class":"tf.keras.layers.Dense",
      "kwargs": {"units":1,"activation":"sigmoid"}
    }]

    discriminator = get_stacked_network(hyperparameters)

    # build model
    model = SupervisedAdversarialAutoEncoder(
      encoder=encoder,
      decoder=decoder,
      discriminator=discriminator,
      code_dim=args.code_size)

    batch, label = next(iter(dataset))
    y = model(batch)

    #model.build(input_shape=(None,64,64,1))
    #model._set_inputs((None,64,64,1))
    #model.compute_output_shape(input_shape=(None, 64,64,1))

    # persist hyperparameters to output dir
    all_hyperpars["command_line_params"] = vars(args)
    with open(output_dir + "all_hyperparameters.json","w") as f:
      json.dump(all_hyperpars,f)

  # create optimizer
  if args.optim_dict is None:
    optimizer = tf.keras.optimizers.Adam()
  else:
    with open(args.optim_dict,"r") as f:
      hyperparameters = json.loads(f.read())
    all_hyperpars["optim"] = hyperparameters
    optimizer = getattr(tf.keras.optimizers,hyperparameters["optimizer"]["class"])(**hyperparameters["optimizer"]["kwargs"])

  model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
  # set callbacks
  log_dir = output_dir + "tensorbord"
  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  checkpoint_path = output_dir + "model"
  #Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
  #checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=False,
    save_freq="epoch")

  def scheduler(epoch,lr):
    return lr*float(args.shrink_factor)**epoch

  lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

  callbacks = [tb_callback,cp_callback,lr_callback]

  # # add user defined callbacks
  # if "callbacks" in hyperparameters.keys():
  #   for callback in hyperparameters["callbacks"]:
  #     cb = getattr(tf.keras.callbacks,callback["class"])(**callback["kwargs"])
  #     callbacks.append(cb)

  #train model
  model.fit(dataset, steps_per_epoch = args.steps_per_epoch, epochs = args.n_epochs, callbacks=callbacks)
  
  #set input dimensions in order to save model

  #input = tf.keras.layers.Input(shape=(64,64,1))
  #out = model(input)

  model.save(output_dir)


  #y = model(batch)
  #model.save(output_dir + "model")

  # # compute validation metrics
  # val_dataset = handler.get_evaluation_dataset(folder=args.val_folder,batch_size=32)
  # eval_metrics = dict(zip(["loss","accuracy"],model.evaluate(val_dataset)))

  # with open(output_dir + "validation-metrics.json","w") as f:
  #   json.dump(eval_metrics,f)

  # #### compute confusion matrix
  # predicted_raw = model.predict(val_dataset)
  # predicted_scores = np.max(predicted_raw,axis=1)
  # predicted_labels = np.array([handler.classes[x] for x in np.argmax(predicted_raw,axis=1)])
  # true_labels = np.concatenate([[handler.classes[z] for z in np.argmax(y,axis=1)] for x,y in val_dataset])

  # predmat = confusion_matrix(true_labels,predicted_labels,normalize="true")
  # predmat = pd.DataFrame(predmat)
  # predmat.columns = list(handler.classes)
  # predmat.index = list(handler.classes)

  # imgs = np.empty((0,64+2*args.padding,64+2*args.padding,1))
  # for img, label in val_dataset:
  #   imgs = np.concatenate([imgs,(img.numpy()).astype(np.uint8)],axis=0)

  # # sample misclassified example in validation set for examination
  # print("sampling misclassified elemens for inspection..")
  # inspect_output = output_dir + "inspect"
  # Path(inspect_output).mkdir(parents=True, exist_ok=True)
  # examiner = ValidationDataExaminer(predicted_labels,true_labels,predicted_scores,imgs,inspect_output)
  # examiner.filter_model_results(ascending_order=False,n=100,misclassified_only=True)
  # examiner.filter_model_results(ascending_order=True,n=100,misclassified_only=False)

  # plt.rcParams.update({'font.size': 8})
  # plt.figure(figsize = (20,20))
  # sn.heatmap(predmat, annot=True)
  # plt.savefig(output_dir + "confusion-matrix.png")


if __name__ == "__main__":
  fit_model()