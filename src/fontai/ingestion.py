import os
import time
import requests 
import zipfile
import re
from shutil import copyfile

import numpy as np

import urllib.request
from bs4 import BeautifulSoup

from abc import ABC, abstractmethod

from typing import Generator
#https://www.1001freefonts.com/
#https://www.dafont.com/

class ZipScrapperIngestor:

  def download_url(self, url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

  def create_folder(self,folder: str):
    if not os.path.isdir(folder):
      os.mkdir(folder)

  def scrape(self,input_url: str,output_path: str):
    #self.download_url(input_url,output_path)
    try:
      self.download_url(input_url,output_path)
      #unzip_file(output_path,unzip_path)
    except Exception as e:
      print("error: {e}".format(e=e))
  
  def ingest(self,urls: Generator[str,None,None], output_folder):
    self.create_folder(output_folder)
    for url in urls:
      #print("url: {u}".format(u=url))
      filename = url.split("/")[-1 if url[-1] != "/" else -2]
      output_path = output_folder + "/" + filename
      #print(output_path)
      self.scrape(url,output_path)

class UrlGenerator(ABC):

  @abstractmethod
  def get_urls(self,**kwargs) -> Generator[str,None,None]:
    pass

class GoogleFontsUrlGenerator(UrlGenerator):

  def get_urls(self,**kwargs) -> Generator[str,None,None]:
    yield "https://github.com/google/fonts/archive/master.zip"

class FreeFontsUrlGenerator(UrlGenerator):

  def get_urls(self,min_id=20,max_id=28000):
    
    for font_id in range(min_id,max_id):
      fid = str(font_id)
      print("downloading id {id}...".format(id=fid))
      font_url = "https://www.1001freefonts.com/d/{id}/".format(id=fid)
      yield font_url

class DafontsUrlGenerator(UrlGenerator):

  def get_urls(self):
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
