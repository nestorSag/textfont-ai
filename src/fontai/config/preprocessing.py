from pathlib import Path
import logging
import typing as t
import inspect
import string
from argparse import Namespace

from pydantic import BaseModel, PositiveInt
import strictyaml as yml

from fontai.core import DataPath
from fontai.cnfig.core import BaseConfigHandler


logger = logging.getLogger(__name__)

class Config(BaseModel):
  """
  Wrapper class for the configuration of the ImageExtractor class

  output_path: folder in which scrapped and zipped ttf/otf files will be saved

  max_zip_size: maximum pre-compression size of zipped output files

  scrappers: list of FileScrapper instances from which scrapped files will be processed.

  """
  input_path: DataPath
  output_path: DataPath
  output_array_size: int
  font_to_array_config: FontToArrayConfig
  beam_parameters: Namespace
  yaml: yml.YAML

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True



class FontToArrayConfig(BaseModel):

  charset: str
  font_size: PositiveInt
  canvas_size: PositiveInt
  canvas_padding: PositiveInt

  def to_dict(self):

    return {
      "charset": charset,
      "font_size": font_size
      "canvas_size": canvas_size,
      "canvas_padding": canvas_padding
    }


class ConfigHandler(object):
  """
  Wrapper for ingestion's configuration processing logic.

  """

  def __init__(self):
    
    self.CONFIG_SCHEMA = yml.Map({
      "output_path": yml.Str(), 
      "input_path": yml.str(), 
      "output_array_size": yml.int(), 
      "font_extraction_size": yml.int(), 
      "font_canvas_size": yml.int(), 
      "font_canvas_padding": yml.int(), 
      yml.Optional("charset", default = string.ascii_letters + string.digits): yml.Str(),
      yml.Optional("beam_parameters", default = {"runner": "direct"}): yml.MapPattern(
            yml.Str(),
            yml.Int() | yml.Float() | yml.Str() | yml.Bool())
       })

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path, input_path, output_array_size, font_extraction_size, font_canvas_size, font_canvas_padding, beam_parameters = DataPath(config.data["output_path"]), config.data["max_zip_size"], config.data["scrappers"]

    output_path = DataPath(config.data["output_path"])
    input_path = DataPath(config.data["input_path"])
    output_array_size = config.data["output_array_size"]
    font_extraction_size = config.data["font_extraction_size"]
    charset = "".join(set(config.data["charset"]))
    font_canvas_size = config.data["font_canvas_size"]
    font_canvas_padding = config.data["font_canvas_padding"]
    beam_parameters = Namespace(**config.data["beam_parameters"])


    f2a_config = FontToArrayConfig(
      charset = charset,
      font_size = font_size,
      canvas_size = canvas_size,
      canvas_padding = canvas_padding
      )


    if f2a_config.canvas_padding >=  f2a_config.canvas_size:
      raise ValueError(f"canvas padding value ({f2a_config.canvas_padding}) is too large for canvas size ({f2a_config.canvas_size})")

    return Config(
      output_path = output_path, 
      input_path = input_path, 
      output_array_size = output_array_size,
      font_to_array_config = f2a_config,
      beam_parameters = beam_parameters,
      yaml = config)
