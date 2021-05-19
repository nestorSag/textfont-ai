from abc import ABC
import logging
import typing as t
import pickle

from fontai.core.base import BaseConfigHandler, MLPipelineStage
from fontai.config.pipeline import Config
from fontai.core.io import DataPath


class MLPipeline(object):

  def __init__(self, config: Config):
    self.initial_config = config
    self.runtime_config = config

  def run_from_config(self):

    for stage in self.pipeline:

      stage.run_from_config()

  def process(self, input_data):
    pass
  def create_pipeline_from_config(self):
    pass
  def process_batch(self, input_path: DataPath, output_folder: DataPath):
    pass

    
