import typing as t

import strictyaml as yml

from fontai.config.core import BaseConfigHandler, BasePipelineTransformConfig

import fontai.pipeline.stage as stages

logger = logging.getLogger(__name__)


class Config(BasePipelineTransformConfig):

  stages: t.List[type]
  configs: t.List[BasePipelineTransformConfig]

class ConfigHandler(BaseConfigHandler):
  
  def get_config_schema(self):
    
    schema = yml.Map({
      "stages" : yml.Seq(yml.Map(
        "class": yml.Str(),
        "yaml_config_path": yml.Str()))
    })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    stages = [getattr(stages, stage.get("class").text) for stage in config.get("stages")]
    configs = [getattr(stages, stage.get("class").text).parse_config_file(stage.get("yaml_config_path").text) for stage in config.get("stages")]

    return Config(
      stages = stages,
      configs = configs,
      yaml = config)
