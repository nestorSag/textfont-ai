import typing as t
import logging

import strictyaml as yml

from fontai.config.core import BaseConfigHandler, BasePipelineTransformConfig

import fontai.runners.stages as stages

logger = logging.getLogger(__name__)


class Config(BasePipelineTransformConfig):

  """Configuration object for Pipeline stage instances

  Attributes:
      stages (t.List[type]): List with types of pipeline stages
      configs: (t.List[BasePipelineTransformConfig]): List with parsed configuration objects for each stage
  """

  stages: t.List[type]
  configs: t.List[BasePipelineTransformConfig]
  fit_stage: t.List[bool]

class ConfigHandler(BaseConfigHandler):
  
  STAGE_TYPES = {
    "ingestion": stages.Ingestion,
    "preprocessing": stages.Preprocessing,
    "scoring": stages.Scoring
  }


  def get_config_schema(self):
    
    schema = yml.Map({
      "stages" : yml.Seq(yml.Map({
        "type": yml.Str(),
        yml.Optional("fit", default=False): yml.Bool(),
        "config": yml.Any() #configs will be matched to a schema later
        })
      )
    })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.
        
    Args:
        config (yml.YAML): YAML object from the strictyaml library
    
    Returns:
        Config: Instantiated configuration
    
    """

    stages = []
    configs = []
    fit_stage = []
    for entry in config.get("stages"):
      stages.append(self.STAGE_TYPES[entry.get("type").text])
      configs.append(entry.get("config"))
      fit_stage.append(entry.get("fit"))

    configs = [stage.parse_config_file(cfg) for stage, cfg in zip(stages, configs)]
    #configs = [getattr(stages, stage.get("class").text).parse_config_file(stage.get("yaml_config_path").text) for stage in config.get("stages")]

    return Config(
      stages = stages,
      configs = configs,
      fit_stage = fit,
      yaml = config)
