"""This module provides an abstraction fo the storage layer in order to read or write bytestreams to different media. Currently, read/writes are supported for local storage and GCS, and reads are also supported for URLs


"""

from __future__ import annotations
from pathlib import Path
import io
import zipfile
import sys
import re
import typing as t
import requests
from abc import ABC, abstractmethod
import logging


from apache_beam.io.gcp.gcsio import GcsIO

from numpy import ndarray

logger = logging.getLogger(__name__)


class BytestreamHandler(ABC):
  """This class provides an interface to underlying storage media

  """
  
  @abstractmethod
  def read(self, path: str) -> bytes:
    """Reads the byte contents from a file
    
    Args:
        path (str): path to file

    Returns:
        bytes object
    """
    pass

  @abstractmethod
  def write(self, path: str, content: bytes) -> None:
    """Writes a bytestream to storage
    
    Args:
        path (str): Description
        content (bytes): Description
    """
    pass

  @abstractmethod
  def list_sources(self, path: str) -> t.Generator[str, None, None]:
    """List files in the folder that path points to
    
    Args:
        path (str): Target folder

    Returns:
        A generator containing string paths to all sources inside target folder

    """
    pass


class LocalBytestreamHandler(BytestreamHandler):
  """Class to interface with local storage
  """
  
  def read(self, path: str) -> bytes:
    path = Path(path)
    if path.is_file():
      #return InMemoryBytestream(name=path.name, content=path.read_bytes())
      return path.read_bytes()
    else:
      raise Exception(f"Path ({str(path)}) does not point to file")
      #raise Exception(f"An error occurred while trying to read {str(path)}: {e}")

  def write(self, path: str, content: bytes) -> None:
    path = Path(path)
    Path(path.parent).mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

  def list_sources(self, path: str):
    path = Path(path)
    contents = path.iterdir()
    return (str(content) for content in contents if content.is_file())


class GcsBytestreamHandler(BytestreamHandler):

  """Class to interface with Google Cloud Storage.
  """
  
  def read(self, url: str) -> bytes:
    gcs_file = GcsIO().open(url,mode="r")
    content = gcs_file.read()
    gcs_file.close()
    return content

  def write(self, url: str, content: bytes) -> None:
    gcs_file = GcsIO().open(url,mode="w")
    gcs_file.write(content)
    gcs_file.close()

  def list_sources(self,url: str) -> t.List[str]:
    #url = self.as_str(url) 
    raw_list = list(GcsIO().list_prefix(url).keys())

    return (elem for elem in raw_list if Path(elem) != Path(url))


class UrlBytestreamHandler(BytestreamHandler):

  """Class to download files from URLs
  """
  
  def read(self, url: str) ->bytes:
    r = requests.get(url, stream=True)
    bf = io.BytesIO()
    with io.BytesIO() as bf:
      for chunk in r.iter_content(chunk_size=1024*1024):
        bf.write(chunk)
      content = bf.getvalue()
    return content

  def write(self, url: str, content: bytes):
    raise NotImplementedError("Bytestreams cannot be written to a url address")

  def list_sources(self,url: str) -> t.List[str]:
    raise NotImplementedError("Bytestreams cannot be listed from a url address")


class BytestreamHandlerFactory(object):

  """Factory method that determines the appropriate file handler class based on the string path
  
  """
  allowed_prefixes = {
    "gs://": GcsBytestreamHandler,
    "https://": UrlBytestreamHandler,
    "http://": UrlBytestreamHandler
  }

  @classmethod
  def create(cls, path: str):
    """Creates an appropriate BystreamHandler instance for the storage medium referenced in path; defaults to local storage if no match is found for remote storage media.
    
    Args:
        path (str): Path to storage location
    
    Returns:
        BytestreamHandler: storage interface for the matched storage medium.
    
    """

    for prefix in cls.allowed_prefixes:
      if prefix in path:
        return cls.allowed_prefixes[prefix]()

    return LocalBytestreamHandler()


class BytestreamPath(object):
  """
    Data reader/writer class that abstracts the underlying storage location. Supports local and GCS storage and downloadable URLs


  """

  def __init__(self, source_str: str):
    """
    
    Args:
        source_str (str): target storage location
    """
    self.string = str(source_str)

    #extensions
    self.is_gcs = "gs://" in self.string
    self.is_http = "http://" in self.string
    self.is_https = "https://" in self.string

    self.handler = BytestreamHandlerFactory.create(self.string)

  @property
  def filename(self):
    if self.is_url():
      filename = self.string.split("/")[-1]
    else:
      filename = Path(self.string).name
    return filename

  def is_url(self):
    """Returns a boolean 
    
    """
    return self.is_gcs or self.is_http or self.is_https

  def extend_url_path(self, suffix: str) -> BytestreamPath:
    """Appends a suffix to the instance's storage path 
    
    Args:
        suffix (str): suffix. Usually a filename.
    
    Returns:
        BytestreamPath: BytestreamPath pointing to the suffixed storage path
    
    Raises:
        ValueError: If no remote storage is matched to the instance's path
    """
    def extend(preffix, string, suffix):
      suffixed = string.replace(preffix,"") + "/" + suffix
      suffixed = re.sub("/+","/",suffixed)
      return BytestreamPath(preffix + suffixed)

    if self.is_gcs:
      return extend("gs://", self.string, suffix)
    elif self.is_http:
      return extend("http://", self.string, suffix)
    elif self.is_https:
      return extend("https://", self.string, suffix)
    else:
      raise ValueError("url does not match any valid preffix.")

  def read_bytes(self) -> bytes:
    """
      Reads bystream from the path

      Returns :
          the file's bytestream
    """
    return self.handler.read(self.string)

  def write_bytes(self,content: bytes) -> None:
    """
      Writes bytestream to path
    """

    self.handler.write(self.string, content)

  def list_sources(self) -> t.Generator[BytestreamPath, None, None]:
    """
      List files (but not dirs) in the folder given by the instance's storage path

      Returns a generator of BytestreamPath objects corresponding to each source file.
    """

    for elem in self.handler.list_sources(self.string):
      yield BytestreamPath(elem)

  def __truediv__(self, path: str) -> BytestreamPath:
    if not isinstance(path, str):
      raise TypeError("path must be a string")
    elif self.is_url():
      return self.extend_url_path(path)
    else:
      return BytestreamPath(str(Path(self.string) / path))

  def __str__(self):
    return self.string