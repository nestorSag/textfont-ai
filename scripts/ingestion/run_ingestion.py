import sys
import argparse
import logging
from pathlib import Path

from fontai.ingestion.downloader import Ingestor

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

def ingest_data(args):
  # Run ingestion pipeline as defiend by the passed configuration YAML file
  
  parser = argparse.ArgumentParser(description = "Scraps free text font files from a configured source.")
  parser.add_argument(
      '--config-file',
      dest='config',
      required = True,      
      help=
      """
      path to YAML file that defines the execution of the ingestion pipeline.
      config schema:
      -----
      output_folder: str
      max_zip_size: float (positive) -> max pre-compression size of resulting zip chunk files
      retrievers: list[{'class': 'className', 'kwargs': {...}}] -> list of FontScrapper subclasses and their kwargs that will be used as sources.
      -----
      """)
  # parser.add_argument(
  #   '--logging-output',
  #   dest='logging_output',
  #   required=True,
  #   help='Logging output file inside logs/ dir')

  logging.basicConfig(filename=Path("logs") / "ingestion.log", level=logging.DEBUG)

  args, _ = parser.parse_known_args(args)
  
  ingestor = Ingestor(ConfigHandler.from_path(Path(args.config)))
  ingestor.run()

if __name__ == "__main__":

  ingest_data(sys.argv)

# example run: python scripts/ingestion/run_ingestion.py --config config/parameters/ingestion/config.yaml