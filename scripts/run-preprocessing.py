from src.processing import *

data_folder = "test"
sources = [data_folder + "/" + subdir + "/zip" for subdir in ["google","1001free","dafont"]]
sinks = [source.replace("zip","ttf") for source in sources]

#unzip
# g_unzipper = GoogleZipExtractor()
# unzipper = ZipExtractor()

# for source, sink in zip(sources,sinks):
#   for filename in os.listdir(source):
#     filepath = source + "/" + filename
#     if "google" in source:
#       g_unzipper.process(filepath,sink)
#     else:
#       unzipper.process(filepath,sink)

sources = sinks
png_sink = data_folder + "/processed/raw_pngs"

# extractor = PngExtractor()

# for source in sinks:
#   for filename in os.listdir(source):
#     if filename.split(".")[-1] == "ttf":
#       #print(filename)
#       extractor.process(source + "/" + filename,png_sink)

normaliser = PngNormaliser()

box_dim = normaliser.get_global_bounding_box(png_sink)