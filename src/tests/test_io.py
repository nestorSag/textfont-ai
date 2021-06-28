import pytest
import zipfile
import io

from PIL import ImageFont

from fontai.io.storage import BytestreamPath
from fontai.io.formats import InMemoryFile, InMemoryZipHolder, InMemoryFontfileHolder, InMemoryZipBundler
from fontai.io.records import LabeledChar, ScoredLabeledChar, LabeledFont, ScoredLabeledFont
from fontai.io.readers import ReaderClassFactory, TfrReader, FileReader, ZipReader, FontfileReader
from fontai.io.writers import WriterClassFactory, TfrWriter, ZipWriter, FileWriter

from tensorflow import constant as tf_constant, convert_to_tensor, Tensor
from tensorflow.train import Example as TFExample
from tensorflow.data import TFRecordDataset


from numpy import empty as np_empty , ones as np_ones, zeros as np_zeros, uint8 as np_uint8, array as np_array

TEST_DATA_PATHS = [
("src/tests/data/io/input", "src/tests/data/io/output"),
("gs://textfont-ai-data/test/input","gs://textfont-ai-data/test/output")]

TEST_ZIP_FOLDER = "src/tests/data/io/formats/zip"
TEST_TTF_FOLDER = "src/tests/data/io/formats/ttf"

TEST_ZIPFILE = "src/tests/data/io/formats/zip/achilles"
TEST_TTF_FILE = "src/tests/data/io/formats/ttf/0AFE_Jen.ttf"


charset_tensor = convert_to_tensor(np_array(list("ab")))

dummy_char_features = 255 * np_ones((64,64), dtype=np_uint8)
dummy_char_label = "a"
dummy_char_score = np_array([0.3, 0.7])

dummy_font_features = 255 * np_ones((2,64,64), dtype=np_uint8)
dummy_font_label = np_array(["a","b"])
dummy_font_scores = np_array([[0.3, 0.7],[0.5,0.5]])

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


@pytest.mark.parametrize("record_class, record_args, charset",[
  (
    LabeledChar, 
    {"features": dummy_char_features, "label": dummy_char_label, "fontname": "dummyfont"}, 
    charset_tensor
  ),
  (
    ScoredLabeledChar, 
    {
      "example": LabeledChar(features = dummy_char_features, label = dummy_char_label, fontname =  "dummyfont"), 
      "score": dummy_char_score},
    charset_tensor
  ),
  (
    LabeledFont, 
    {"features": dummy_font_features, "label": dummy_font_label, "fontname": "dummyfont"}, 
    charset_tensor
  ),
  (
    ScoredLabeledFont, 
    {
      "example": LabeledFont(features = dummy_font_features, label =  dummy_font_label, fontname = "dummyfont"),
      "score": dummy_font_scores}, 
    charset_tensor
  )
  ])
def test_records(record_class, record_args, charset):

  # create record instance
  sample_record = record_class(**record_args)

  assert isinstance(sample_record.to_tf_example(), TFExample)
  
  # serialise and deserialise
  serialised = sample_record.to_tf_example().SerializeToString()
  bytes_dict = record_class.from_tf_example(convert_to_tensor(serialised))
  #bytes_dict = sample_rcord.parse_bytes_dict(bytes_dict)

  # parse for model training
  # features, label = record_class.parse_dict_for_training(
  #   bytes_dict, 
  #   charset_tensor = charset_tensor)
  parsed_dict = record_class.parse_bytes_dict(bytes_dict)
  features, label = record_class.get_training_parser(charset_tensor=charset_tensor)(parsed_dict)
  
  assert isinstance(features, Tensor)
  assert isinstance(label, Tensor)
  assert len(label) == len(charset)
  #assert features.shape == sample_record.features.shape

  # recover original record instance
  #record = record_class.parse_dict_for_scoring(bytes_dict)
  parsed_dict = record_class.parse_bytes_dict(record_class.from_tf_example(convert_to_tensor(serialised)))
  record = record_class.get_scoring_parser(charset_tensor=charset_tensor)(parsed_dict)
  assert isinstance(record, record_class)

@pytest.mark.parametrize("input_path, output_path", TEST_DATA_PATHS)
def test_storage_interface(input_path, output_path):

  input_path = BytestreamPath(input_path)
  output_path = BytestreamPath(output_path)

  assert isinstance(input_path / "testfile.txt", BytestreamPath)
  assert [str(elem) for elem in input_path.list_sources()] == [str(input_path / "testfile.txt")]
  assert str((input_path / "testfile.txt").read_bytes().decode("utf-8")) == "this is a test"

  (output_path / "write_test.txt").write_bytes(bytes("this is a write test".encode("utf-8")))
  assert str((output_path / "write_test.txt").read_bytes().decode("utf-8")) == "this is a write test"


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