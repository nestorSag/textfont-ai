# example run: python scripts/ingestion/run_ingestion.py --config-file config/parameters/google-fonts-ingestion.yaml
import sys
import argparse
import logging
from pathlib import Path

from fontai.pipeline.stages import FontIngestion

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=Path("logs") / "ingestion.log", level=logging.INFO, filemode = "w")

def ingest_data(args):
  # Run ingestion pipeline as defined by the passed configuration YAML file
  
  parser = argparse.ArgumentParser(description = "Scraps free text font files from a configured source.")
  parser.add_argument(
      '--config-file',
      dest='config_file',
      required = True,      
      help=
      """
      path to YAML file that defines the execution of the ingestion pipeline.
      """)

  args, _ = parser.parse_known_args(args)
  
  print(f"Redirecting logs to logs/ingestion.log")

  FontIngestion.run_from_config_file(args.config_file)

if __name__ == "__main__":

  ingest_data(sys.argv)

