import pytest
import zipfile
import io

from PIL import ImageFont

from fontai.io.storage import BytestreamPath
from fontai.io.formats import InMemoryFile, InMemoryZipHolder, InMemoryFontfileHolder, InMemoryZipBundler
from fontai.io.records import LabeledExample, ScoredExample, ScoredLabeledExample
from fontai.io.readers import ReaderClassFactory, TfrReader, FileReader, ZipReader, FontfileReader
from fontai.io.writers import WriterClassFactory, TfrWriter, ZipWriter, FileWriter

from tensorflow import constant as tf_constant
from tensorflow.train import Example as TFExample
from tensorflow.data import TFRecordDataset


from numpy import empty as np_empty 

TEST_DATA_PATHS = [
("src/tests/data/io/input", "src/tests/data/io/output"),
("gs://textfont-ai-data/test/input","gs://textfont-ai-data/test/output")]

TEST_ZIP_FOLDER = "src/tests/data/io/formats/zip"
TEST_TTF_FOLDER = "src/tests/data/io/formats/ttf"

TEST_ZIPFILE = "src/tests/data/io/formats/zip/achilles"
TEST_TTF_FILE = "src/tests/data/io/formats/ttf/0AFE_Jen.ttf"


@pytest.mark.parametrize("input_path, output_path", TEST_DATA_PATHS)
def test_storage_interface(input_path, output_path):

  input_path = BytestreamPath(input_path)
  output_path = BytestreamPath(output_path)

  assert isinstance(input_path / "testfile.txt", BytestreamPath)
  assert [str(elem) for elem in input_path.list_sources()] == [str(input_path / "testfile.txt")]
  assert str((input_path / "testfile.txt").read_bytes().decode("utf-8")) == "this is a test"

  (output_path / "write_test.txt").write_bytes(bytes("this is a write test".encode("utf-8")))
  assert str((output_path / "write_test.txt").read_bytes().decode("utf-8")) == "this is a write test"

def test_formats():
  
  zipped = InMemoryFile(filename="holder", content = BytestreamPath(TEST_ZIPFILE).read_bytes())\
  .to_format(InMemoryZipHolder)\
  .deserialise()

  assert isinstance(zipped, zipfile.ZipFile)

  ttf = InMemoryFile(filename="holder", content = BytestreamPath(TEST_TTF_FILE).read_bytes())\
  .to_format(InMemoryFontfileHolder)\
  .deserialise(font_size=10)

  assert isinstance(ttf, ImageFont.FreeTypeFont)

  zipped_files = zipped.namelist()

  bundler = InMemoryZipBundler()
  for file in zipped_files:
    bundler.write(InMemoryFile(filename=file, content = zipped.read(file)))

  from_bundle = zipfile.ZipFile(io.BytesIO(bundler.compress().get_bytes()))
  assert from_bundle.namelist() == zipped_files



def test_records():

  sample_record = ScoredExample(features = np_empty((32,32)), score = np_empty((4,)))

  assert isinstance(sample_record.to_tfr(), TFExample)

  sample_record = LabeledExample(features = np_empty((32,32)), label = "a", fontname = "my font")

  assert isinstance(sample_record.to_tfr(), TFExample)

  sample_record = ScoredLabeledExample(labeled_example = sample_record, score = tf.convert_to_tensor(np_empty((4,))))

  assert isinstance(sample_record.to_tfr(), TFExample)

def test_readers():
  reader_class = ReaderClassFactory.get(TFRecordDataset)
  assert reader_class == TfrReader

  reader_class = ReaderClassFactory.get(InMemoryFile)
  assert reader_class == FileReader

  reader_class = ReaderClassFactory.get(InMemoryZipHolder)
  assert reader_class == ZipReader

  reader = reader_class(TEST_ZIP_FOLDER)
  for file in reader.get_files():
    assert isinstance(file.deserialise(), zipfile.ZipFile)

  reader_class = ReaderClassFactory.get(InMemoryFontfileHolder)
  assert reader_class == FontfileReader

  reader = reader_class(TEST_TTF_FOLDER)
  for file in reader.get_files():
    assert isinstance(file.deserialise(font_size=10), ImageFont.FreeTypeFont)



def test_writers():
  writer_class = WriterClassFactory.get(TFRecordDataset)
  assert writer_class == TfrWriter

  writer_class = WriterClassFactory.get(InMemoryFile)
  assert writer_class == FileWriter