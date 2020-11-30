from __future__ import absolute_import

import argparse
import logging

from past.builtins import unicode

import apache_beam as beam
import zipfile
from PIL import Image, ImageFont, ImageDraw
import io
from datetime import datetime

from apache_beam.io.gcp.gcsio import GcsIO
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

def get_char_dim(gray_img):
  nonzero = np.where(gray_img > 0)
  if nonzero[0].shape == (0,) or nonzero[1].shape == (0,):
    return (0, 0), (0,0)
  else:
    dims = [(np.min(axis),np.max(axis)) for axis in nonzero]
    return dims

def unzip(gcs_file):
  try:
    yield zipfile.ZipFile(io.BytesIO(gcs_file.read()))
  except zipfile.BadZipFile as e:
    return

def get_ttfs(zip,font_size = 100, png_size=(500,500),offset=50):
  filenames = [filename.lower() for filename in zip.namelist()]
  # avoid unrelated files and bold and italic fonts
  valid = [filename for file in files if ".ttf" in filename and "italic.ttf" not in filename and "bold.ttf" not in filename and "light.ttf" not in filename]
  for file in valid:
    for letter in string.ascii_letters + string.digits:
      try:
        ttf = zip.read(file)
        im = Image.new("RGB",png_size)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(io.BytesIO(ttf),font_size)
        draw.text((offset,offset),letter,font=font)
        # this filename is to avoid overwriting files
        output_filename = letter + "@" + file.split("/")[-1].replace(".ttf","") + str(datetime.now().time()) + ".png"
        yield file, im
      except Exception as e:
        return 

def write_and_summarise(tuple,output_folder):
  # prepare to save to gcs
  output_filename, im = tuple
  outfile = GcsIO().open(output_folder + "/" + output_filename,mode="w")
  bf = io.BytesIO()
  im.save(bf,format="png")
  # get bounding box dimension
  gray_img = np.mean(imageio.imread(bf.getvalue(),format="png"),axis=-1).astype(np.uint8)
  h_bound, w_bound = get_char_dim(gray_img)
  h = h_bound[1] - h_bound[0] + 1
  w = w_bound[1] - w_bound[0] + 1
  #save cropped png
  cropped = gray_img[h_bound[0]:(h_bound[0] + h),w_bound[0]:(w_bound[0]+w)]
  bf.close()
  bf = io.BytesIO()
  imageio.imwrite(bf,im=cropped,format="png")
  outfile.write(bf.getvalue())
  outfile.close()
  return (1,(h,w))

def get_elementwise_max(tuples):
  return [max([elem[0] for elem in tuples]),max([elem[1] for elem in tuples])]

def run(argv=None, save_main_session=True):
  """Main entry point; defines and runs the wordcount pipeline."""

  parser = argparse.ArgumentParser(description = "processes font ZIP files into individual characters' PNG files")
  parser.add_argument(
      '--input',
      dest='input',
      required = True,
      help='Input file to process.')
  parser.add_argument(
      '--output',
      dest='output',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      required = True,      
      help='Output file to write results to.')
  parser.add_argument(
      '--png-size',
      dest='temp_png_size',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default='500',
      help='Dim of PNG output in which characters will be embedded. Needs to be large enough for provided font size')
  parser.add_argument(
      '--font-size',
      dest='font_size',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default='100',
      help='Font size to use when extracting PNGs from TTFs')
  parser.add_argument(
      '--offset',
      dest='offset',
      # CHANGE 1/6: The Google Cloud Storage path is required
      # for outputting the results.
      default='128',
      help='Offset from top left corner of PNG when embedding the characters PNG; barroque fonts can overflow bounding PNG if offset is too small or too large.')

  known_args, pipeline_args = parser.parse_known_args(argv)
  pipeline_args.extend([
      # CHANGE 2/6: (OPTIONAL) Change this to DataflowRunner to
      # run your pipeline on the Google Cloud Dataflow Service.
      '--runner=DirectRunner',
      # CHANGE 3/6: (OPTIONAL) Your project ID is required in order to
      # run your pipeline on the Google Cloud Dataflow Service.
      '--project=SET_YOUR_PROJECT_ID_HERE',
      # CHANGE 4/6: (OPTIONAL) The Google Cloud region (e.g. us-central1)
      # is required in order to run your pipeline on the Google Cloud
      # Dataflow Service.
      '--region=SET_REGION_HERE',
      # CHANGE 5/6: Your Google Cloud Storage path is required for staging local
      # files.
      '--staging_location=gs://YOUR_BUCKET_NAME/AND_STAGING_DIRECTORY',
      # CHANGE 6/6: Your Google Cloud Storage path is required for temporary
      # files.
      '--temp_location=gs://YOUR_BUCKET_NAME/AND_TEMP_DIRECTORY',
      '--job_name=textfont-ai-zip2png',
  ])

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  with beam.Pipeline(options=pipeline_options) as p:

    # Read the text file[pattern] into a PCollection.
    zips = p | GcsIO().open(known_args.input,model="r")

    # Count the occurrences of each word.
    pngs = (
        zips
        | 'unzip' >> beam.Map(lambda x: unzip(x))
        | 'getTTFs' >> beam.FlatMap(lambda x: get_ttfs(x))
        | 'saveAndSummarise' >> beam.Map(lambda x: write_and_summarise(x)))
        | 'GetGlobalBoundingBox' >> beam.combine(get_elementwise_max)

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()