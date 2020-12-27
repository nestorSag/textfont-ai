from fontai.ingestion import *

data_folder = "raw"
sources = [GoogleFontsUrlGenerator(),FreefontsUrlGenerator(),DafontsUrlGenerator()]
sinks = [data_folder + "/" subdir + "/zip" for subdir in ["google","1001free","dafont"]]

scrapper = ZipScrapperIngestor()

for (source,sink) in zip(sources,sinks):
  scrapper.ingest(source.get_urls(),sink)