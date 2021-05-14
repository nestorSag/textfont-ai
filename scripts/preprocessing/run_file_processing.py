import sys
import argparse
import logging
from pathlib import Path

from fontai.preprocessing.file_preprocessing import FileProcessor
from fontai.config.preprocessing import ConfigHandler

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

def process_files(args):
  # Run file processing Beam pipeline as defined by the passed configuration YAML file
  
  parser = argparse.ArgumentParser(description = "Maps output from ingestion stage to tensorflow Record files for model consumption.")
  parser.add_argument(
      '--config-file',
      dest='config',
      required = True,      
      help=
      """
      path to YAML file that defines the execution of the file processing pipeline.
      """)

  logging.basicConfig(filename=Path("logs") / "file-processing.log", level=logging.DEBUG, filemode = "w")

  args, _ = parser.parse_known_args(args)
  
  processor = FileProcessor(ConfigHandler().from_file(Path(args.config)))
  processor.run()

if __name__ == "__main__":

  process_files(sys.argv)

# example run: python scripts/ingestion/run_ingestion.py --config config/parameters/ingestion/config.yaml