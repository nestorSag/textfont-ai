import pytest
from pathlib import Path

from fontai.core import DataPath

TEST_DATA_PATHS = [
("src/tests/data/core/input", "src/tests/data/core/output"),
("gs://textfont-ai-data/test/input","gs://textfont-ai-data/test/output")]

@pytest.mark.parametrize("input_path, output_path", TEST_DATA_PATHS)
def test_data_path(input_path, output_path):

  input_path = DataPath(input_path)
  output_path = DataPath(output_path)

  assert isinstance(input_path / "testfile.txt", DataPath)
  assert [str(elem) for elem in input_path.list_files()] == [str(input_path / "testfile.txt")]
  assert str((input_path / "testfile.txt").read_bytes().decode("utf-8")) == "this is a test"

  (output_path / "write_test.txt").write_bytes(bytes("this is a write test".encode("utf-8")))
  assert str((output_path / "write_test.txt").read_bytes().decode("utf-8")) == "this is a write test"