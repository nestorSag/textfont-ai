import sys
import argparse
import logging
from pathlib import Path

from fontai.ingestion.downloader import Ingestor
from fontai.config.ingestion import ConfigHandler

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

def ingest_data(args):
  # Run ingestion pipeline as defined by the passed configuration YAML file
  
  parser = argparse.ArgumentParser(description = "Scraps free text font files from a configured source.")
  parser.add_argument(
      '--config-file',
      dest='config',
      required = True,      
      help=
      """
      path to YAML file that defines the execution of the ingestion pipeline.
      """)

  logging.basicConfig(filename=Path("logs") / "ingestion.log", level=logging.DEBUG, filemode = "w")

  args, _ = parser.parse_known_args(args)
  
  ingestor = Ingestor(ConfigHandler().from_file(Path(args.config)))
  ingestor.run()

if __name__ == "__main__":

  ingest_data(sys.argv)

# example run: python scripts/ingestion/run_ingestion.py --config-file config/parameters/ingestion/config.yaml