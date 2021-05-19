from pathlib import Path
import logging
import typing as t
import inspect
import string
from argparse import Namespace

from pydantic import BaseModel, PositiveInt
import strictyaml as yml

from fontai.core.base import BaseConfigHandler, BaseConfig


logger = logging.getLogger(__name__)


class FontExtractionConfig(BaseModel):
  """
  Data class that holds the runtime parameters to extract image arrays from files

  charset: string containing characters to be extracted from font files

  font_extraction_size: Font size to use when conveting fonts to images

  canvas_size: Height and width of buffer array in which fonts will be extracted before being processed further

  canvas_padding: Padding used in the canvas array when extracting the fonts

  """

  charset: str
  font_extraction_size: PositiveInt
  canvas_size: PositiveInt
  canvas_padding: PositiveInt

  def as_dict(self):

    return {
      "charset": self.charset,
      "font_extraction_size": self.font_extraction_size,
      "canvas_size": self.canvas_size,
      "canvas_padding": self.canvas_padding
    }

  @classmethod
  def from_yaml(cls, yaml: yml.YAML):
    return FontExtractionConfig(**yaml.data)


class Config(BaseConfig):
  """
  Configuration class for the image extraction pipeline stage

  output_array_size: size of the final grayscale image corresponding to each font's characters

  font_to_array_config: Data object with runtime parameters for exctracting image arrays from files

  beam_cmd_line_args: List of command line arguments passed to the Beam pipeline

  """
  output_array_size: PositiveInt
  font_to_array_config: FontExtractionConfig
  beam_cmd_line_args: t.List[str]

    
class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for image processing stage's configuration handling logic

  """

  def get_config_schema(self):
    
    schema = yml.Map({
      "output_path": self.IO_CONFIG_SCHEMA, 
      "input_path": self.IO_CONFIG_SCHEMA,
      "output_array_size": yml.Int(),
      "font_extraction_config": yml.Map({
        "font_extraction_size": yml.Int(), 
        "canvas_size": yml.Int(), 
        "canvas_padding": yml.Int(),
        yml.Optional("charset", default = string.ascii_letters + string.digits): yml.Str()
        })
      yml.Optional("beam_cmd_line_args", default = ["--runner", "DirectRunner"]): yml.Seq(yml.Str())
       })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path = self.instantiate_io_handler(config.get("output_path"))
    input_path = self.instantiate_io_handler(config.get("input_path"))
    beam_cmd_line_args = config.data["beam_cmd_line_args"]
    output_array_size = config.get("output_array_size").data
    f2a_config = FontExtractionConfig.from_yaml(**config.get("font_extraction_config").data)


    if f2a_config.canvas_padding >=  f2a_config.canvas_size/2:
      raise ValueError(f"canvas padding value ({f2a_config.canvas_padding}) is too large for canvas size ({f2a_config.canvas_size})")

    return Config(
      output_path = output_path, 
      input_path = input_path, 
      output_array_size = output_array_size,
      font_to_array_config = f2a_config,
      beam_cmd_line_args = beam_cmd_line_args,
      yaml = config)
