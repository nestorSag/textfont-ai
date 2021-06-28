from pathlib import Path
import logging
import typing as t
import inspect
import string
from argparse import Namespace

from pydantic import BaseModel, PositiveInt, PositiveFloat, validator
import strictyaml as yml

from fontai.config.core import BaseConfigHandler, BasePipelineTransformConfig
import fontai.io.records as records

logger = logging.getLogger(__name__)


class FontExtractionConfig(BaseModel):
  """
  Data class that holds the runtime parameters to extract image arrays from files
  
  Args:
      charset (str): string containing characters to be extracted from font files
      font_extraction_size (int): Font size to use when conveting fonts to images
      canvas_size (int): Height and width of buffer array in which fonts will be extracted before being processed further
      canvas_padding (int): Padding used in the canvas array when extracting the fonts
  
  """

  charset: str
  font_extraction_size: PositiveInt
  canvas_size: PositiveInt
  canvas_padding: PositiveInt


class Config(BasePipelineTransformConfig):
  """
  Configuration class for the image extraction pipeline stage
  
  Args:
      output_record_class (records.TfrWritable): tfr-compatible output classes from the module `fontai.io.records`; currently supported are `LabeledChar` and `LabeledFont`
      output_array_size (int): size of the final grayscale image corresponding to each font's characters
      font_to_array_config (int): Data object with runtime parameters for exctracting image arrays from files
      beam_cmd_line_args (t.List[str]): List of command line arguments passed to the Beam pipeline

  """
  output_record_class: type
  output_array_size: PositiveInt
  max_output_file_size: PositiveFloat
  font_to_array_config: FontExtractionConfig
  beam_cmd_line_args: t.List[str]

  @validator("output_record_class")
  def validate_output_schema(schema_class):
    """Validate input record class
    
    Args:
        schema_class (type)
    
    Returns:
        type: Validated record class
    
    Raises:
        TypeError: If record class not in allowed set
    """
    supported = [records.LabeledChar, records.LabeledFont]
    if schema_class in supported:
      return schema_class
    else:
      raise TypeError(f"supported output_record_classes are {[x.__name__ for x in supported]}")

    
class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for image processing stage's configuration handling logic

  """

  def get_config_schema(self):
    
    schema = yml.Map({
      "output_record_class": yml.Str(), 
      "output_array_size": yml.Int(),
      "font_extraction_size": yml.Int(), 
      "canvas_size": yml.Int(), 
      "canvas_padding": yml.Int(),
      yml.Optional("input_path", default = None): self.IO_CONFIG_SCHEMA, 
      yml.Optional("output_path", default = None): self.IO_CONFIG_SCHEMA,
      yml.Optional("charset", default = string.ascii_letters + string.digits): yml.Str(),
      yml.Optional("max_output_file_size", default = 64.0): yml.Float(),
      yml.Optional("beam_cmd_line_args", default = ["--runner", "DirectRunner"]): yml.Seq(yml.Str())
       })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    Args:
        config: YAML object from the strictyaml library

    """
    
    input_path, output_path = config.get("input_path").text, config.get("output_path").text
    output_record_class = getattr(records, config.get("output_record_class").text)
    beam_cmd_line_args = config.data["beam_cmd_line_args"]
    output_array_size = config.get("output_array_size").data
    max_output_file_size = config.get("max_output_file_size").data
    f2a_config = FontExtractionConfig(
      charset = config.get("charset").data,
      font_extraction_size = config.get("font_extraction_size").data,
      canvas_size = config.get("canvas_size").data,
      canvas_padding = config.get("canvas_padding").data)


    if f2a_config.canvas_padding >=  f2a_config.canvas_size/2:
      raise ValueError(f"canvas padding value ({f2a_config.canvas_padding}) is too large for canvas size ({f2a_config.canvas_size})")


    logger.info(f"Setting output schema as {output_record_class.__name__}")
    
    return Config(
      output_record_class = output_record_class,
      input_path = input_path, 
      output_path = output_path, 
      output_array_size = output_array_size,
      max_output_file_size = max_output_file_size,
      font_to_array_config = f2a_config,
      beam_cmd_line_args = beam_cmd_line_args,
      yaml = config)
