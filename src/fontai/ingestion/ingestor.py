"""This module contains logic to retrieve scrapped files from one or more scrapper instances and persist them to storage.

"""
import typing as t
import logging

from fontai.config.ingestion import Config

logger = logging.getLogger(__name__)

class Ingestor(object):

  """Ingestor class that retrieves zipped font files; it is initialised from a configuration object that defines its execution.
  """

  def __init__(self, config: Config):
    """
    
    Args:
        config (Config): Configuration object
    """
    config = config

  def run_from_config(self) -> None:

    writes_counter = 0
    for scrapper in self.config.scrappers:
      for url in scrapper.get_source_urls():
        try:
          logger.debug(f"Retrieving data from {url}")
          content = url.read_bytes()
          output_path = (self.config.output_path / str(writes_counter))
          logger.debug(f"persisting to {output_path}")
          output_path.write_bytes(content)
          writes_counter += 1
        except Exception as e:
          logger.exception(f"An error ocurred when scrapping {url}: {e}")