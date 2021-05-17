from pathlib import Path
import logging
import typing as t
import inspect
import string
from argparse import Namespace

from pydantic import BaseModel, PositiveInt
import strictyaml as yml

from fontai.core import DataPath
from fontai.config.core import BaseConfigHandler


logger = logging.getLogger(__name__)


class FontExtractionConfig(BaseModel):

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


class Config(BaseModel):
  """
  Configuration class for the image extraction pipeline stage

  input_path: folder from which zipped ttf/otf files will be fetched

  output_path: Folder in which Tensorflow record files will be persisted

  output_array_size: size of the final grayscale image corresponding to each font's characters

  beam_cmd_line_args: List of command line arguments passed to the Beam pipeline

  yaml: original YAML object built from the configuration file contents

  """
  input_path: DataPath
  output_path: DataPath
  output_array_size: PositiveInt
  font_to_array_config: FontExtractionConfig
  beam_cmd_line_args: t.List[str]
  yaml: yml.YAML

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

    
class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for image processing stage's configuration handling logic

  """

  def get_config_schema(self):
    
    schema = yml.Map({
      "output_path": yml.Str(), 
      "input_path": yml.Str(),
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
    output_path = DataPath(config.get("output_path").data)
    input_path = DataPath(config.get("input_path").data)
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
