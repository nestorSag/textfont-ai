import pytest
from pathlib import Path

from fontai.core import DataPath

TEST_DATA_PATHS = [("src/tests/data/core"),("gs://textfont-ai-data/test")]

@pytest.mark.parametrize("path", TEST_DATA_PATHS)
def test_data_path(path):

  path = DataPath(path)

  assert [str(elem) for elem in path.list_files(folder)] == [str(path / "testfile.txt")]
  assert str((path / "testfile.txt").read_bytes().decode("utf-8")) == "this is a test"

  (Path(folder) / "write_test.txt").write_bytes(bytes("this is a write test".encode("utf-8")))
  assert str((path / "write_test.txt").read_bytes().decode("utf-8")) == "this is a write test"