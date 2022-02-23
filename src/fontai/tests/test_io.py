import pytest
import zipfile
import io

from PIL import ImageFont

from fontai.io.storage import BytestreamPath
from fontai.io.formats import InMemoryFile, InMemoryZipfile, InMemoryFontfile
from fontai.io.records import LabeledChar, ScoredLabeledChar, LabeledFont, ScoredLabeledFont
from fontai.io.readers import TfrReader, FileReader, ZipReader, FontfileReader
from fontai.io.writers import TfrWriter, ZipWriter, FileWriter, InMemoryZipBundler

from tensorflow import constant as tf_constant, convert_to_tensor, Tensor, reduce_all
from tensorflow.train import Example as TFExample
from tensorflow.data import TFRecordDataset


from numpy import empty as np_empty , ones as np_ones, zeros as np_zeros, uint8 as np_uint8, array as np_array, float32 as np_float, all as np_all

TEST_DATA_PATHS = [
("src/fontai/tests/data/io/input", "src/fontai/tests/data/io/output")]#,
#("gs://textfont-ai-data/test/local/input","gs://textfont-ai-data/test/local/output")]

TEST_ZIP_FOLDER = "src/fontai/tests/data/io/formats/zip"
TEST_TTF_FOLDER = "src/fontai/tests/data/io/formats/ttf"

TEST_ZIPFILE = "src/fontai/tests/data/io/formats/zip/achilles"
TEST_TTF_FILE = "src/fontai/tests/data/io/formats/ttf/0AFE_Jen.ttf"

charset = "ab"

charset_tensor = np_array([str.encode(x) for x in list(charset)])

score = convert_to_tensor(np_empty((len(charset),), dtype=np_float))

dummy_char_features = 255 * np_ones((64,64), dtype=np_uint8)
dummy_char_label = "a"
dummy_char_score = np_array([0.3, 0.7])

dummy_font_features = 255 * np_ones((2,64,64), dtype=np_uint8)
dummy_font_label = np_array(["a","b"])
dummy_font_scores = np_array([[0.3, 0.7],[0.5,0.5]])

def test_formats():
  
  zipped = InMemoryFile(filename="holder", content = BytestreamPath(TEST_ZIPFILE).read_bytes())\
  .to_format(InMemoryZipfile)\
  .deserialise()

  assert isinstance(zipped, zipfile.ZipFile)

  ttf = InMemoryFile(filename="holder", content = BytestreamPath(TEST_TTF_FILE).read_bytes())\
  .to_format(InMemoryFontfile)\
  .deserialise(font_size=10)

  assert isinstance(ttf, ImageFont.FreeTypeFont)


@pytest.mark.parametrize("record_class, record_args, charset_tensor, score",[
  (
    LabeledChar, 
    {"features": dummy_char_features, "label": dummy_char_label, "fontname": "dummyfont"}, 
    charset_tensor,
    score
  ),
  (
    LabeledFont, 
    {"features": dummy_font_features, "label": dummy_font_label, "fontname": "dummyfont"}, 
    charset_tensor,
    score
  )
  ])
def test_records(record_class, record_args, charset_tensor, score):

  # create record instance
  sample_record = record_class(**record_args)

  assert isinstance(sample_record.to_tf_example(), TFExample)
  
  # serialise and deserialise
  serialised = sample_record.to_tf_example().SerializeToString()
  bytes_dict = record_class.from_tf_example(convert_to_tensor(serialised))
  #bytes_dict = sample_rcord.parse_bytes_dict(bytes_dict)

  parsed_dict = record_class.parse_bytes_dict(bytes_dict)
  features, label = record_class.get_training_parser(charset_tensor=charset_tensor)(parsed_dict)
  
  assert isinstance(features, Tensor)
  assert isinstance(label, Tensor)
  assert len(label) == len(charset)
  #assert features.shape == sample_record.features.shape

  # recover original record instance
  parsed_dict = record_class.parse_bytes_dict(record_class.from_tf_example(convert_to_tensor(serialised)))
  recovered = record_class.from_parsed_bytes_dict(parsed_dict)
  assert isinstance(recovered, sample_record.__class__)

  ## test scored records
  scored_record = sample_record.add_score(score=score.numpy(), charset_tensor=charset_tensor)
  serialised = scored_record.to_tf_example().SerializeToString()
  parsed_dict = scored_record.parse_bytes_dict(scored_record.from_tf_example(convert_to_tensor(serialised)))
  recovered = scored_record.from_parsed_bytes_dict(parsed_dict)
  recovered.example.features = (255 * recovered.example.features).astype(np_uint8)
  # cannot try directly recovered == scored_record because when a font example is serialised and deserialised, the features are 4-dimensional (instead of 3-dimensional) to comply with TF training shape requirements
  assert isinstance(recovered, scored_record.__class__)
  assert np_all(recovered.score == scored_record.score)
  assert reduce_all(recovered.charset_tensor == scored_record.charset_tensor)

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
  reader = ZipReader(TEST_ZIP_FOLDER)
  for file in reader.get_files():
    assert isinstance(file.deserialise(), zipfile.ZipFile)

  reader = FontfileReader(TEST_TTF_FOLDER)
  for file in reader.get_files():
    assert isinstance(file.deserialise(font_size=10), ImageFont.FreeTypeFont)

def test_writers():
  zipped = InMemoryFile(filename="holder", content = BytestreamPath(TEST_ZIPFILE).read_bytes())\
  .to_format(InMemoryZipfile)

  zipped_content = zipped.content

  zipped = zipped.deserialise()
  zipped_files = zipped.namelist()

  bundler = InMemoryZipBundler()
  for file in zipped_files:
    bundler.write(InMemoryFile(filename=file, content = zipped.read(file)))
  bundler.compress()
  reopened = zipfile.ZipFile(io.BytesIO(bundler.get_bytes()), "r")
  bundler.close()
  assert True

  