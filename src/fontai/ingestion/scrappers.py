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

from fontai.core.base import InMemoryFile

logger = logging.getLogger(__name__)

# google fonts source: "https://github.com/google/fonts/archive/main.zip"

class FreeFontsFileScrapper(DataPath):
  """
  Font scrapper for https://www.1001freefonts.com

  min_id: minimum font id to attempt to retrieve

  max_id: minimum font id to attempt to retrieve

  """

  def __init__(self):
    super().__init__("www.1001freefonts.com")
    self.min_id = 0
    self.max_id = 27000

  def list_files(self):
    
    for font_id in range(self.min_id,self.max_id):
      font_url = f"https://www.1001freefonts.com/d/{font_id}/"
      yield DataPath(font_url)


class DafontsFileScrapper(DataPath):
  """
    Font scrapper for https://www.dafont.com/

  """
  def __init__(self):
    super().__init__("www.dafont.com")


  def list_files(self):
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
        logger.exception("an error occured while scrapping website (there mya be a single scrappable link) {e}".format(e=e))
        n_pages = 1
      for page_number in range(1,n_pages+1):
        page = "alpha.php?lettre={l}&page={k}&fpp=200".format(l=letter.lower(),k=page_number)
        if not ((letter == "A" and page_number in list(range(1,11)) + [20]) or (letter == "B" and page_number in list(range(1,11)) + [24])):
          logger.info("downloading page {p}".format(p=page))
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


class MultiSourceFileScrapper(DataPath):

  def __init__(self, paths: t.List[str]):

    self.string = "Multi-source scrapper instance"

    def str_to_data_path(src):
      if src == "www.dafont.com":
        return DafontsFileScrapper()
      elif src == "www.1001freefonts.com":
        return FreeFontsDataPath()
      else:
        return DataPath(src)

    self.sources = [str_to_data_path(path) for path in paths]
    self.error_on_io = "This instance is meant to be a scrapper and does not implement reading or writing methods; only list_files() is implemented"

  def list_files(self):

    for src in self.sources:
      for sub_src in src.list_files():
        yield DataPath(sub_src)

  def write_bytes(self, content: bytes):

    raise NotImplementedException(self.error_on_io)

  def read_bytes(self):

    raise NotImplementedException(self.error_on_io)


