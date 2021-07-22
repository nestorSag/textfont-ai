"""
  This module contains the definitions of high-level ML lifecycle stage classes; at the moment this includes ingestion, preprocessing and training/scoring.
"""

import logging
from collections import OrderedDict
from pathlib import Path
import typing
import traceback
import os
import string
import sys
import signal
import typing as t

from PIL import ImageFont
from numpy import ndarray
from strictyaml import as_document

from tensorflow import Tensor
import tensorflow as tf
from tensorflow.data import TFRecordDataset


import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from fontai.config.preprocessing import Config as ProcessingConfig, ConfigHandler as ProcessingConfigHandler
from fontai.config.ingestion import Config as IngestionConfig, ConfigHandler as IngestionConfigHandler
from fontai.config.prediction import ModelFactory, TrainingConfig, Config as ScoringConfig, ConfigHandler as ScoringConfigHandler
from fontai.config.deployment import Config as DeploymentConfig, ConfigHandler as DeploymentConfigHandler, Grid

from fontai.prediction.input_processing import RecordPreprocessor
from fontai.preprocessing.mappings import PipelineFactory, BeamCompatibleWrapper, Writer, ZipToFontFiles

#from fontai.runners.base import MLPipelineTransform
from fontai.io.storage import BytestreamPath
from fontai.io.readers import ScrapperReader, ZipReader, TfrReader
from fontai.io.writers import TfrWriter, ZipWriter
from fontai.io.records import TfrWritable
from fontai.io.formats import InMemoryFile, InMemoryZipfile
from fontai.runners.base import ConfigurableTransform, IdentityTransform, FittableTransform

from fontai.deployment.visualise import *

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from numpy import array as np_array, clip as np_clip

import mlflow 

logger = logging.Logger(__name__)
  
__all__ = [
  "Ingestion",
  "Preprocessing",
  "Scoring",
  "Deployment"
  ]
  
class Ingestion(ConfigurableTransform, IdentityTransform):

  """Retrieves zipped font files. It takes a list of scrappers defined in `fontai.io.scrappers` from which it downloads files to storage.
  
  """

  def __init__(self, config: IngestionConfig = None):
    
    self.config = config

  @classmethod
  def get_config_parser(cls):
    return IngestionConfigHandler()

  @classmethod
  def from_config_object(cls, config: IngestionConfig, **kwargs):
    return cls(config)

  @classmethod
  def run_from_config_object(cls, config: IngestionConfig):
    
    ingestor = cls.from_config_object(config)
    font_extractor = ZipToFontFiles()
    writer = ZipWriter(ingestor.config.output_path, config.max_output_file_size)
    for file in ScrapperReader(config.scrappers).get_files():
      for font_file in font_extractor.map(file):
        writer.write(font_file)

  @classmethod
  def get_stage_name(cls):
    return "ingestion"

  def transform_batch(self, input_path: str, output_path: str):
    raise NotImplementedError("This method is not implemented for ingestion.")


class Preprocessing(ConfigurableTransform):
  """
  Processes zipped font files and outputs Tensorflow records consisting of labeled images for ML consumption.

  """

  def __init__(
    self, 
    output_record_class: type,
    charset: str,
    font_extraction_size: int,
    canvas_size: int,
    canvas_padding: int,
    output_array_size: int,
    beam_cmd_line_args: t.List[str] = []):
    """
    
    Args:
        output_record_class (type): Output schema class, inheriting from TfrWritable
        charset (str): String with characters to be extracted
        font_extraction_size (int): Font size to use when extracting font images
        canvas_size (int): Image canvas size in which fonts will be extracted
        canvas_padding (int): Padding in the image extraction canvas
        output_array_size (int): Final character image size
        beam_cmd_line_args (t.List[str], optional): List of Apache Beam command line arguments for distributed processing
    """
    self.beam_cmd_line_args = beam_cmd_line_args

    self.pipeline = PipelineFactory.create(
      output_record_class = output_record_class,
      charset = charset,
      font_extraction_size = font_extraction_size,
      canvas_size = canvas_size,
      canvas_padding = canvas_padding,
      output_array_size = output_array_size)


  @classmethod
  def from_config_object(cls, config: ProcessingConfig, **kwargs):

    return cls(
      output_record_class = config.output_record_class,
      output_array_size = config.output_array_size,
      beam_cmd_line_args = config.beam_cmd_line_args,
      **config.font_to_array_config.dict())

    return cls()

  def transform(self, data):
    return self.pipeline.map(data)

  @classmethod
  def get_config_parser(cls):
    return ProcessingConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "preprocessing"

  @classmethod
  def run_from_config_object(cls, config: ProcessingConfig):

    # set provisional value for CUDA env variable to prevent out of memory errors from the GPU; this occurs because the preprocessing code depends on (CPU-bound) Tensorflow functionality, which attempts to seize memory from the GPU automatically, but this throws an error when Beam uses multiple threads.
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    output_path, input_path = config.output_path, config.input_path

    processor = cls.from_config_object(config)

    # if output is locally persisted, create parent folders
    if not BytestreamPath(output_path).is_url():
      Path(str(output_path)).mkdir(parents=True, exist_ok=True)

    pipeline_options = PipelineOptions(processor.beam_cmd_line_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:

      # execute pipeline
      (p
      | 'create source list' >> beam.Create(BytestreamPath(input_path).list_sources()) #create list of sources as strings
      | 'read zipped files' >> beam.Map(lambda filepath: InMemoryZipfile.from_bytestream_path(filepath)) #line needed to load files lazily and not overwhelm memory
      | 'get labeled exampes from zip files' >> beam.ParDo(
        BeamCompatibleWrapper(
          mapper = PipelineFactory.create(
            output_record_class = config.output_record_class,
            output_array_size = config.output_array_size,
            **config.font_to_array_config.dict())
        )
      )
      | "write to storage" >> beam.ParDo(Writer(TfrWriter(output_path, config.max_output_file_size))))

    # unset provisional value for env variable
    if visible_devices is None:
      del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
      os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices



class Scoring(FittableTransform):
  """
  Trains a prediction model or uses one to score input data.
  

  Attributes:
      model (keras.Model): Scoring model
      CHARSET_OPTIONS (t.Dict): Dictionary from allowed charsets names to charsets
      training_config (TrainingConfig): training configuration wrapper
      charset_tensor (Tensor): Tensor with an entry per character in the current charset
  
  """
  CHARSET_OPTIONS = {
    "uppercase": string.ascii_letters[26::],
    "lowercase": string.ascii_letters[0:26],
    "digits": string.digits,
    "all": string.ascii_letters + string.digits
    }

  def __init__(
    self, 
    model: tf.keras.Model, 
    training_config: TrainingConfig = None, 
    charset: str = "lowercase"):
    """
    
    Args:
        model (tf.keras.Model): Scoring model
        training_config (TrainingConfig, optional): Training configuration wrapper
        charset (str): charset to use for training and batch scoring. It must be one of 'lowercase', 'uppercase' or 'digits', or otherwise a string with all characters under consideration
    """
    self.model = model
    self.training_config = training_config
    self.charset = charset


    try:
      self.charset = self.CHARSET_OPTIONS[charset]
    except KeyError as e:
      logger.warning(f"Charset string is not one from {list(self.CHARSET_OPTIONS.keys())}; creating custom charset from provided string instead.")
      self.charset = "".join(list(set(charset)))

    self.num_classes = len(self.charset)
    self.charset_tensor = np_array([str.encode(x) for x in list(self.charset)])


  def fit(self, data: TFRecordDataset):
    """Fits the scoring model with the passed data
    
    Args:
        data (TFRecordDataset): training data
    
    Returns:
        Scoring: Scoring with trained model
    
    Raises:
        ValueError: If training_config is None (not provided).
    """

    if self.training_config is None:
      raise ValueError("Training configuration not provided at instantiation time.")
    
    self.model.compile(
      loss = self.training_config.loss, 
      optimizer = self.training_config.optimizer,
      metrics = self.training_config.metrics,
      run_eagerly=False)

    self.model.fit(
      data,
      #training_data,
      steps_per_epoch = self.training_config.steps_per_epoch, 
      epochs = self.training_config.epochs,
      callbacks = self.training_config.callbacks)

    return self

  def _to_shape(self, x: t.Union[ndarray, Tensor]):
    """Reshape single example to be transformed in-memory by the `transform` method.
    
    Args:
        x (t.Union[ndarray, Tensor]): Single input
    
    Returns:
        t.Union[ndarray, Tensor]: Reshaped input
    """
    if len(x.shape) == 2:
      x = x.reshape((1,) + x.shape + (1,))
    elif len(x.shape) == 3:
      x = x.reshape((1,) + x.shape)
    return x

  def transform(self, input_data: t.Union[ndarray, Tensor, TfrWritable]) -> t.Union[Tensor, ndarray, TfrWritable]:
    """
    Process a single instance
    
    Args:
        input_data (t.Union[ndarray, Tensor, TfrWritable]): Input instance
    
    Returns:
        t.Union[Tensor, ndarray, TfrWritable]: Scored example in the corresponding format, depending on the input type.
    
    Raises:
        TypeError: If input type is not allowed.
    
    """

    if isinstance(input_data, (ndarray, Tensor)):
      return self.model.predict(self._to_shape(input_data))
    elif isinstance(input_data, TfrWritable):
      return input_data.add_score(
        score = self.model.predict(self._to_shape(input_data.features)), 
        charset_tensor=self.charset_tensor)
    else:
      raise TypeError("Input type is not one of ndarray, Tensor or TfrWritable")

  @classmethod
  def get_config_parser(cls):
    return ScoringConfigHandler()

  @classmethod
  def get_stage_name(cls):
    return "scoring"


  @classmethod
  def from_config_object(cls, config: ScoringConfig, load_from_model_path = False):
    """Initialises a Scoring instance from a configuration object
    
    Args:
        config (ScoringConfig): COnfiguration object
        load_from_model_path (bool, optional): If True, the model is loaded from the model_path argument in the configuration object.
    
    Returns:
        Scoring: Instantiated Scoring object.
    """
    if load_from_model_path:
      model_class_name = config.model.__class__.__name__
      classname_tuple = ("custom_class", model_class_name if model_class_name != "Sequential" else None)
      
      # dict -> YAML -> Model
      input_dict = {"path": config.model_path}
      if model_class_name != "Sequential":
        input_dict["custom_class"] = model_class_name
      model_yaml = as_document(input_dict)

      logger.info(f"load_from_model_path flag set to False; loading model from model_path {config.model_path} of class {model_class_name}")
      model = ModelFactory().from_yaml(model_yaml)
    else:
      model = config.model

    predictor = Scoring(model = model, training_config = config.training_config, charset = config.charset)
    return predictor

  @classmethod
  def run_from_config_object(cls, config: ScoringConfig, load_from_model_path = False):

    predictor = cls.from_config_object(config, load_from_model_path)

    data_fetcher = RecordPreprocessor(
      input_record_class = config.input_record_class,
      charset_tensor = predictor.charset_tensor,
      custom_filters = predictor.training_config.custom_filters,
      custom_mappers = predictor.training_config.custom_mappers)

    writer = TfrWriter(config.output_path)

    files = TfrReader(config.input_path).get_files()
    data = data_fetcher.fetch(files, training_format=False, batch_size = predictor.training_config.batch_size)

    counter = 0
    for features, labels, fontnames in data:
      try:
        scores = predictor.transform(features)
        
        scored_records = config.input_record_class.from_scored_batch(
          features = features.numpy(),
          labels = labels.numpy(),
          fontnames = fontnames.numpy(), 
          scores = scores,
          charset_tensor = predictor.charset_tensor)

        for record in scored_records:
          writer.write(record)
        counter += 1
      except Exception as e:
        logger.exception(f"Exception scoring batch with features: {features}. Full trace: {traceback.format_exc()}")
      logger.info(f"Processed {counter} examples.")

  @classmethod
  def fit_from_config_object(cls, config: ScoringConfig, load_from_model_path = False, run_id: str = None):
    
    # implement graceful interruptions
    def save_on_sigint(sig, frame):
      predictor.model.save(config.model_path)
      logger.info(f"Training stopped by SIGINT: saving current model to {config.model_path}")
      sys.exit(0)
      
    signal.signal(signal.SIGINT, save_on_sigint)

    # start MLFlow run
    with mlflow.start_run(run_id=run_id, nested=False) as run:

      logger.info(f"MLFlow run id: {run.info.run_id}")

      # log run configuration into MLFlow
      cfg_log_path = "run-configs"
      # check whether there are previous run configs
      client = mlflow.tracking.MlflowClient()
      n_previous_runs = len(client.list_artifacts(run.info.run_id, cfg_log_path))
      current_run = f"{n_previous_runs + 1}.yaml"

      with open(current_run,"w") as f:
        f.write(config.yaml.as_yaml())
      mlflow.log_artifact(current_run,cfg_log_path)
      os.remove(current_run)

      mlflow.tensorflow.autolog() #start keras autologging

      # fetch data and fit model
      predictor = cls.from_config_object(config, load_from_model_path)

      data_fetcher = RecordPreprocessor(
        input_record_class = config.input_record_class,
        charset_tensor = predictor.charset_tensor,
        custom_filters = predictor.training_config.custom_filters,
        custom_mappers = predictor.training_config.custom_mappers)

      data = TfrReader(config.input_path).get_files()

      predictor.fit(data_fetcher.fetch(data, training_format=True, batch_size=predictor.training_config.batch_size))
      logger.info(f"Saving trained model to {config.model_path}")
      predictor.model.save(config.model_path)


class Deployment(ConfigurableTransform):

  """Deploys a small Dash app in which a fitted typeface generative model's style space can be explored

  """
  
  def __init__(
    self,
    model: tf.keras.Model, 
    sampler: t.Callable, 
    grid: Grid, 
    charset_size: int):
    """
    
    Args:
        model (tf.keras.Model): Generative model
        sampler (t.Callable): Style vectors' sampling function
        grid (Grid): Grid describing the style space's grid to explore
        charset_size (int): Number of characters in font
    """
    self.model = model
    self.sampler = sampler
    self.grid = grid
    self.charset_size = charset_size

    self.external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

  def launch(self,**kwargs):
    """Launches a Dash app
    
    Args:
        **kwargs: Additional arguments passed to Dash app
    """
    app = dash.Dash("fontai-deployment", external_stylesheets=self.external_stylesheets)

    step = (self.grid.largest - self.grid.lowest)/self.grid.size
    middle = (self.grid.largest + self.grid.lowest)/2

    app.layout = html.Div(children=[
      html.Div(children=[html.Button('Random', id='button')] +
        [dcc.Slider(min=self.grid.lowest,max=self.grid.largest,step=step, value=middle, id=f"slider-{k}") for k in range(self.grid.dim)],
        style = {'display': 'inline-block', 'width': '25%'}),
      html.Img(id="font_figure")
      ])

    @app.callback(
      [Output(f"slider-{k}","value") for k in range(self.grid.dim)],
      Input("button","n_clicks"))
    def update_random(n_clicks):
      return list(np_clip(self.sampler(size=self.grid.dim), a_min=self.grid.lowest, a_max=self.grid.largest))

    @app.callback(
      Output('font_figure', 'src'),
      [Input(f"slider-{k}", 'value') for k in range(self.grid.dim)])
    def update(*args):
      style_vector = np_array(args).reshape((1,-1))
      font = generate_font(self.model, style_vector, self.charset_size)
      img = plot_font(font)
      return fig_to_str(img)

    app.run_server(**kwargs)

  def process(self, x):
    raise NotImplementedError("This method is not implemented for deployment.")

  @classmethod
  def get_config_parser(cls):
    return DeploymentConfigHandler()

  @classmethod
  def from_config_object(cls, config: DeploymentConfig, **kwargs):
    return cls(
      model = config.model,
      sampler = config.sampler,
      grid = config.grid,
      charset_size = config.charset_size)

  @classmethod
  def run_from_config_object(cls, config: DeploymentConfig):
    
    cls.from_config_object(config).launch(debug=True)

  @classmethod
  def get_stage_name(cls):
    return "deployment"

  def transform_batch(self, input_path: str, output_path: str):
    raise NotImplementedError("This method is not implemented for deployment.")

  @classmethod
  def get_stage_name(cls):
    return "deployment"


