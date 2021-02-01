from fontai.ingestion import *

import argparse

def ingest_data(args):
  parser = argparse.ArgumentParser(description = "Scraps free text fonts from the internet.")
  parser.add_argument(
      '--output-folder',
      dest='output_folder',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default="data/raw",
      required = True,      
      help='Output folder where zip file will be saved')

  args, _ = parser.parse_known_args(argv)
  
  data_folder = args.output_folder
  sources = [GoogleFontsUrlGenerator(),FreefontsUrlGenerator(),DafontsUrlGenerator()]
  sinks = [data_folder + "/" subdir + "/zip" for subdir in ["google","1001free","dafont"]]

  scrapper = ZipScrapperIngestor()

  for (source,sink) in zip(sources,sinks):
    scrapper.ingest(source.get_urls(),sink)

if __name__ == "__main__":

  ingest(sys.argv)