from pathlib import Path

import fontai.io.scrappers as scrapper_module

from fontai.pipeline.stages import FontIngestion

import logging

TEST_INGESTION_CONFIG = """
scrappers:
- class: LocalScrapper
  kwargs: 
    folder: src/tests/data/ingestion/input
output_path: src/tests/data/ingestion/output
"""

logger = logging.getLogger(__name__)

def test_ingestion():
  config = FontIngestion.parse_config_str(TEST_INGESTION_CONFIG)
  FontIngestion.run_from_config_object(config)

  assert [obj.name for obj in list(Path(config.output_path).iterdir())] == [obj.name for obj in list(Path(config.scrappers[0].folder).iterdir())]



