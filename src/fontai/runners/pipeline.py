"""
  This module contains a basic orchestrator for the execution of sequential data transformation stages.
"""
from __future__ import annotations
import typing as t
import types
from fontai.config.pipeline import Config as PipelineConfig, ConfigHandler as PipelineConfigHandler
from fontai.runners.base import ConfigurableTransform, FittableTransform
from fontai.config.core import BasePipelineTransformConfig


class ManyToManyTransform(object):

  """Helper class to execute one-to-many many-to-many transformations in the pipeline
  
  Attributes:
      core_transform (ConfigurableTransform): Core transformer class
  """

  def __init__(self, core_transform):
    self.core_transform = core_transform

  def transform(self, data: t.Any):
    """Outputs a generator of transformed elements
    
    Args:
        data (t.Any): Input data
    
    Yields:
        t.Any: individual outputs
    """
    for elem in self.to_generator(data):
      for out in self.to_generator(self.core_transform.transform(elem)):
        yield out

  def to_generator(self, data):
    if not isinstance(data, types.GeneratorType):
      return iter((data,))
    else:
      return data

class Pipeline(ConfigurableTransform):

  """Pipeline class to execute a sequence of ConfigurableTransforms; this allows to perform the whole set of transformations from raw data to (possible multiple) trained models
  
  Attributes:
      streaming_pipeline (t.List[ConfigurableTransform]): List of instantiated transforms
      transforms (type): classes of pipeline stages inheriting from ConfigurableTransform. Possible choices are defined in the fontai.runners.stages module
      configs (t.List[BasePipelineTransformConfig]): Sequence of configuration files to instantiate and execute each stage
      fit_stage (t.List[bool]): If True, fit the corresponding pipeline stage instead of using it for scoring. It is ignored if the stage is not fittable.
  """

  def __init__(self, transforms: t.List[type], configs: t.List[BasePipelineTransformConfig], fit_stage: t.List[bool]):
    """Summary
    
    Args:
        transforms (t.List[type]): List of transformations in the pipeline
        configs (t.List[BasePipelineTransformConfig]): List of parsed configurations, one per stage in the pipeline
        fit_stage (t.List[bool]): If True, fit the corresponding pipeline stage instead of using it for scoring. It is ignored if the stage is not fittable.
    """
    self.transforms = transforms
    self.configs = configs
    self.fit_stage = fit_stage

    self.streaming_pipeline = [
      ManyToManyTransform(core_transform = transform.from_config_object(config)) for transform, config in zip(self.transforms, self.configs)]

  def transform(self, data: t.Any) -> t.Any:
    
    out = data
    for streaming_transform in self.streaming_pipeline:
      out = streaming_transform.transform(out)

    return out

  @classmethod
  def from_config_object(cls, config: PipelineConfig) -> Pipeline:
    return cls(config.stages, config.configs, config.fit_stage)

  @classmethod
  def run_from_config_object(cls, config: PipelineConfig) -> None:
    pipeline = cls.from_config_object(config)
    for transform, config, fit in zip(pipeline.transforms, pipeline.configs, pipeline.fit_stage):
      if fit and issubclass(transform, FittableTransform):
        transform.fit_from_config_object(config)
      else:
        transform.run_from_config_object(config)

  @classmethod
  def get_config_parser(cls) -> PipelineConfigHandler:
    return PipelineConfigHandler()