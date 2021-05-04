import os
import time
import io
import requests 
import zipfile
import re
from shutil import copyfile
import typing as t
from abc import ABC, abstractmethod
import urllib.request
import logging
from pathlib import Path


import numpy as np

from bs4 import BeautifulSoup

from pydantic import BaseModel


logger = logging.getLogger(__name__)

class InMemoryFile(BaseModel):
  # wrapper that holds the bytestreams of unzipped in-memory ttf/otf files
  filename: str
  content: bytes

class FileScrapper(ABC):
  """
     Interface implemented by specialised web-scrapping subclasses to retrieve zipped font files.
  """

  @abstractmethod
  def get_source_string(self) -> str:
    """
    Prints the online resource from which fonts are being retrieved
    """
    pass

  @abstractmethod
  def get_sources(self) -> t.Generator[t.Union[str,Path],None,None]:
    """
    Generator method that yields urls for all scrappable fonts in the current website
    """
    pass

  def get_stream_from_source(self,url: t.Union[str,Path]) -> bytes:
    """
    Generator method that retrieves streams from each scrapped url; this default implementation assumes a single stream per url.
    """
    logger.info(f"target url: {url}")

    r = requests.get(url, stream=True)
    bf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=chunk_size):
      bf.write(chunk)
    return bf.getvalue()

  def get_stream_tuples(self) -> t.Tuple[str,t.Generator[bytes, None, None]]:
    """
    Generator method that yields all the scrappable streams from a website and their sources.

    kwargs: args to be passed to the get_sources() method
    """
    for url in self.get_sources():
      yield str(url), self.get_stream_from_source(url)

  def unpack_files_from_stream(self, stream: bytes, source: t.Optional[str] = None) -> t.Generator[InMemoryFile,None,None]:
    """
    Generator method that yields all font files from a zip bytestream

    stream: bytestream from in-memory zip file

    source: name from source file, for exception logging.

    Returns tuples of the form (file bytestream, zip filename)

    """

    if source is None:
      source = "unspecified"

    def choose_ext(lst):
      ttfs = len([x for x in lst if ".ttf" in x.lower()])
      otfs = len([x for x in lst if ".otf" in x.lower()])
      if ttfs >= otfs:
        return ".ttf"
      else:
        return ".otf"

    #we assume the stream is a zip file's contents
    try:
      zipped = zipfile.ZipFile(io.BytesIO(stream))
    except Exception as e:
      logger.exception(f"Error: source ({source}) can't be read as zip")
      return
    files_in_zip = zipped.namelist()
    # choose whether to proces TTFs or OTFs, but not both
    ext = choose_ext(files_in_zip)
    valid_files = sorted([filename for filename in files_in_zip if ext in filename.lower()])
    
    for file in valid_files:
      filename = Path(file).name
      try: 
        content = zipped.read(file)
        yield InMemoryFile(filename=filename, content = content)
      except Exception as e:
        logger.exception(f"Error while extracting file {filename} from zip")

  def get_files(self) -> t.Generator[InMemoryFile,None,None]:
    """
    Generator method that yields all scrappable font files (either .ttf or .otf) from the source website

    kwargs: args to be passed to the get_stream_tuples() method

    """
    for source, stream in self.get_stream_tuples():
      for file in self.unpack_files_from_stream(stream, source=source):
        yield file

# FileScrapper sublcasses 


class GoogleFontsFileScrapper(FileScrapper):

  def get_source_string(self):
    return "www.github.com@google@fonts@archive"

  def get_sources(self,**kwargs) -> t.Generator[str,None,None]:
    yield "https://github.com/google/fonts/archive/main.zip"

class FreeFontsFileScrapper(FileScrapper):
  """
  Font scrapper for https://www.1001freefonts.com

  min_id: minimum font id to attempt to retrieve

  max_id: minimum font id to attempt to retrieve

  """

  def __init__(self,min_id,max_id):

    self.min_id = min_id
    self.max_id = max_id

  def get_source_string(self):
    return "www.1001freefonts.com"

  def get_sources(self):
    
    for font_id in range(self.min_id,self.max_id):
      fid = str(font_id)
      print("downloading id {id}...".format(id=fid))
      font_url = "https://www.1001freefonts.com/d/{id}/".format(id=fid)
      yield font_url



class DafontsFileScrapper(FileScrapper):
  """
    Font scrapper for https://www.dafont.com/

  """
  def get_source_string(self):
    return "www.dafont.com"

  def get_sources(self):
    my_ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
      # parse html of first letter page
      letter_page_url = "https://www.dafont.com/alpha.php?lettre={l}&page=1&fpp=200".format(l=letter.lower())

      raw_html = requests.get(letter_page_url,headers = {"user-agent": my_ua}).text
      page_html = BeautifulSoup(raw_html,"html.parser")
      ## find number of letter pages
      letter_pages = page_html.find(class_="noindex").find_all("a")
      page_refs = [page["href"] for page in letter_pages]
      # get number of pages for current letter
      n_pages_rgx = re.compile("page=([0-9]+)")
      try:
      # if this happens, there is a single page
        n_pages = max([int(n_pages_rgx.search(x).group(1)) for x in page_refs])
      except Exception as e:
        print("error: {e}".format(e=e))
        n_pages = 1
      for page_number in range(1,n_pages+1):
        page = "alpha.php?lettre={l}&page={k}&fpp=200".format(l=letter.lower(),k=page_number)
        if True: #not ((letter == "A" and page_number in list(range(1,11)) + [20]) or (letter == "B" and page_number in list(range(1,11)) + [24])):
          print("downloading page {p}".format(p=page))
          page_url = "https://www.dafont.com/" + page.replace("&amp;","")

          raw_html = requests.get(page_url,headers = {"user-agent": my_ua}).text
          #print("raw_html: {x}".format(x=raw_html))
          page_html = BeautifulSoup(raw_html,"html.parser")
          dl_links = page_html.findAll("a",{"class": "dl"})

          #print("dl_links {d}".format(d=dl_links))
          for link in dl_links:
            href = link["href"]
            # random sleep time 
            time.sleep(np.random.uniform(size=1,low=1,high=2)[0])
            yield "https:" + href



class LocalFileScrapper(FileScrapper):
  """
  Emulates the Scrapper class logic for zip files already in local storage. This class is useful for processing again font files that had been downloaded before with a previous code version; source urls have changed since, and some can't be used to crawl the web sources anymore.

  """
  def __init__(self,folder):

    self.folder = Path(folder)
    if not self.folder.is_dir() or len(list(self.folder.iterdir())) == 0:
      raise Exception(f"Folder to be scrapped ({self.folder}) does not exist or is empty.")

  def get_source_string(self):

    return f"local@{self.folder.name}"

  def get_sources(self) -> t.Generator[t.Union[str,Path],None,None]:
    for path in self.folder.iterdir():
      if path.is_file():
        yield path

  def get_stream_from_source(self,path: t.Union[str,Path]) -> bytes:

    return path.read_bytes()