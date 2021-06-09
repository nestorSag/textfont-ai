"""This module contains logic that was used to scrape three font file sources: google, 1001fonts.com and dafont.com. As of May 2021 at least one of those sites have changed their url sources and so, some of these classes might not work anymore, and some work might be required to scrape the files again.

"""
import time
import requests 
import re
import random
import typing as t
from abc import ABC, abstractmethod
import urllib.request
import logging
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

__all__ = [
 "FreeFontsFileScrapper",
 "DafontsFileScrapper",
 "GoogleFontsScrapper"]
 
class Scrapper(ABC):

  """Interface implemented by web scrapper classes. Contains a single method, get_source_urls.
  """

  @abstractmethod
  def get_source_urls(self) -> t.Generator[str, None, None]:
    """Returns a generator of BytestreamPath objects pointing to each scrappable URL

    """
    pass

class GoogleFontsScrapper(Scrapper):

  """Retrieves the main zip file from Google fonts repository
  """
  
  def get_source_urls(self):
    yield "https://github.com/google/fonts/archive/main.zip"


class FreeFontsFileScrapper(Scrapper):

  """Retrieves font files from www.1001freefonts.com
  
  """
  
  def __init__(self):
    self.min_id = 0
    self.max_id = 27000

  def get_source_urls(self):
    
    for font_id in range(self.min_id,self.max_id):
      font_url = f"https://www.1001freefonts.com/d/{font_id}/"
      yield font_url


class DafontsFileScrapper(Scrapper):
  """
    Retrieves font files from www.dafont.com

  """
  def __init__(self):
    super().__init__("www.dafont.com")


  def get_source_urls(self):
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
            time.sleep(random.uniform(1,2))
            yield "https:" + href


class LocalScrapper(Scrapper):

  """
  Scrapper simulator from local files
  
  Attributes:
      folders (t.List[str]): List of source folders
  """
  def __init__(self, folders: t.List[str]):
    self.folders = folders

  def get_source_urls(self):

    for folder in self.folders:
      current = BytestreamPath(folder)
      for file in current.list_sources():
        yield str(file)