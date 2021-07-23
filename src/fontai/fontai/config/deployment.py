from pathlib import Path
import logging
import typing as t
import inspect

from pydantic import PositiveFloat, PositiveInt, BaseModel
import strictyaml as yml

import fontai.io.scrappers as scrapper_module

from fontai.config.core import BaseConfigHandler, BasePipelineTransformConfig
from fontai.config.prediction import ModelFactory

import numpy.random as rnd

import tensorflow as tf

logger = logging.getLogger(__name__)


class Grid(BaseModel):

  """Represents a grid in style space that can be explored through an interactive Dash app
  """

  dim: PositiveInt #dimensionality in style space
  lowest: float #lowest grid value componentwise
  largest: float #largest grid value componentwise
  size: PositiveInt #componentwise grid zie

class Config(BasePipelineTransformConfig):

  model: tf.keras.Model
  sampler: t.Callable
  charset_size: PositiveInt
  grid: Grid
  dash_args: t.Dict

class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for deployment's configuration logic.
  
  """
  @classmethod
  def get_config_schema(self):
    """
    YAML configuration schema:
    
    decoder_path: Path to generative model that takes point in style space and one-hot-encoded character representations and output an image
    charset_size: Dimensionality of the characters' one-hot-encoded representation
    style_vector_size: Dimensionality of style space
    style_sampler (optional, defaults to 'normal'): distribution from numpy.random to sample from style space. 'size' should be its only argument
    min_style_value (optional, defaults to -3): minimum componentwise style value in style space grid
    max_style_value (optional, defaults to 3): maximum componentwise style value in style space grid
    style_grid_size (optional, defaults to 100): Componentwise number of bins in style grid


    """
    
    schema = yml.Map({
      "decoder_path" : self.IO_CONFIG_SCHEMA,
      "charset_size": yml.Int(),
      "style_vector_size": yml.Int(),
      yml.Optional("style_sampler", default="normal"): yml.Str(),
      yml.Optional("min_style_value", default=-3): yml.Float(),
      yml.Optional("max_style_value", default=3): yml.Float(),
      yml.Optional("style_grid_size", default=100): yml.Int(),
      yml.Optional("dash_args", default={}): yml.MapPattern(
        yml.Str(), 
        self.yaml_to_obj.ANY_PRIMITIVES) | yml.EmptyDict()
    })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.
        
    Args:
        config (yml.YAML): YAML object from the strictyaml library
    
    Returns:
        Config: Configuration object
    
    """
    

    
    model = tf.keras.models.load_model(config.get("decoder_path").text)
    sampler = getattr(rnd, config.get("style_sampler").text)
    grid = Grid(
      dim = config.get("style_vector_size").data,
      lowest = config.get("min_style_value").data,
      largest = config.get("max_style_value").data,
      size = config.get("style_grid_size").data)

    dash_args = config.get("dash_args").data

    return Config(
      model = model,
      dash_args = dash_args,
      sampler = sampler,
      charset_size = config.get("charset_size").data,
      grid = grid,
      yaml = config)

  def other_setup(self):
    self.model_factory = ModelFactory()