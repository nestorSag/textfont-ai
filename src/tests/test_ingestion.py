from pathlib import Path

from fontai.pipeline.stages import FontIngestion

TEST_INGESTION_CONFIG = """
scrappers:
- class: LocalScrapper
  kwargs: 
    folders:
    - src/tests/data/ingestion/input
output_path: src/tests/data/ingestion/output
"""

test_config_object = ConfigHandler().from_string(TEST_INGESTION_CONFIG)

def test_ingestion():
  config = FontIngestion.parse_config_str(TEST_INGESTION_CONFIG)
  FontIngestion.run_from_config(config)

  assert Path(config.output_path).iterdir() == Path(config.scrappers[0].folders[0]).iterdir()


