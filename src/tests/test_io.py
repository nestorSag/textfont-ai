import pytest
import zipfile


from PIL import ImageFont

from fontai.io.storage import BytestreamPath
from fontai.io.formats import InMemoryFile, InMemoryZipHolder, InMemoryFontfileHolder, TFDatasetWrapper, InMemoryZipBundler
from fontai.io.records import LabeledExample, ScoredExample
from fontai.io.readers import ReaderClassFactory, TfrReader, BytesReader
from fontai.io.writers import WriterClassFactory, TfrWriter, ZipWriter, BytesWriter


TEST_DATA_PATHS = [
("src/tests/data/io/input", "src/tests/data/io/output"),
("gs://textfont-ai-data/test/input","gs://textfont-ai-data/test/output")]

TEST_ZIP_FOLDER = "src/test/data/io/formats/zip"
TEST_TTF_FOLDER = "src/test/data/io/formats/ttf"

TEST_ZIPFILE = "src/test/data/io/formats/zip/achilles"
TEST_TTF_FILE = "src/test/data/io/formats/ttf/0AFE_Jen.ttf"


@pytest.mark.parametrize("input_path, output_path", TEST_DATA_PATHS)
def test_storage_interface(input_path, output_path):

  input_path = BytestreamPath(input_path)
  output_path = BytestreamPath(output_path)

  assert isinstance(input_path / "testfile.txt", BytestreamPath)
  assert [str(elem) for elem in input_path.list_files()] == [str(input_path / "testfile.txt")]
  assert str((input_path / "testfile.txt").read_bytes().decode("utf-8")) == "this is a test"

  (output_path / "write_test.txt").write_bytes(bytes("this is a write test".encode("utf-8")))
  assert str((output_path / "write_test.txt").read_bytes().decode("utf-8")) == "this is a write test"

def test_formats():
  
  zipped = InMemoryFile(filename="holder", content = BytestreamPath(TEST_ZIPFILE))\
  .to_format(InMemoryZipHolder)\
  .deserialise()

  assert isinstance(zipped, zipfile.ZipFile)

  ttf = InMemoryFile(filename="holder", content = BytestreamPath(TEST_TTF_FILE))\
  .to_format(InMemoryFontfileHolder)\
  .deserialise()

  assert isinstance(ttf, TrueFont.FontType)

  zipped_files = zipped.namelist()

  bundler = InMemoryZipBundler()
  for file in zipped_files:
    bundler.add(InMemoryFile(filename=file, content = zipped.read(file)))

  from_bundle = zipfile.ZipFile(io.BytesIO(bundler.compress().get_bytes()))
  assert from_bundle.namelist() == zipped_files



def test_records():

  sample_record = ScoredExample(features = np.empty((32,32)), score = tf.constant([0.1,.0.2,0.7], dtype = 'float32'))

  assert isinstance(sample_record.to_tfr(), dict)

  sample_record = LabeledExample(features = np.empty((32,32)), label = "a", fontname = "my font")

  assert isinstance(sample_record.to_tfr(), dict)

  sample_record = ScoredLabeledExample(labeled_example = sample_record = np.empty((32,32)), score = tf.constant([0.1,.0.2,0.7], dtype = 'float32'))

  assert isinstance(sample_record.to_tfr(), dict)

def test_readers():
  reader_class = ReaderClassFactory.get(TFDatasetWrapper)
  assert reader_class == TfrWriter

  reader_class = ReaderClassFactory.get(InMemoryFile)
  assert reader_class == BytesReader

  reader_class = ReaderClassFactory.get(InMemoryFontfileHolder)
  assert reader_class == BytesReader

  reader = reader_class(TEST_TTF_FOLDER)
  for file in reader.get_files():
    file.
  reader_class = ReaderClassFactory.get(InMemoryZipBundler)
  assert reader_class == BytesReader




def test_writers():
  pass
