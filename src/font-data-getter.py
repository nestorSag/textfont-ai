import os
import time
import requests 
import zipfile
import re
from shutil import copyfile

import numpy as np

import urllib.request
from bs4 import BeautifulSoup

#https://www.1001freefonts.com/
#https://www.dafont.com/
DATA_DIR = "data/fonts"

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def unzip_file(file,folder):
	with zipfile.ZipFile(file, 'r') as zip_ref:
		zip_ref.extractall(folder)

def collect_ttf_files(source,target):
	for obj in os.listdir(source):
		obj_path = source + "/" + obj
		#print(obj_path)
		if os.path.isdir(obj_path):
			collect_ttf_files(obj_path, target)
		else:
			if bool(re.search(r"Regular\.ttf$",obj_path)):
				copyfile(obj_path,target + "/" + obj)

def get_google_fonts():
	fonts_dir = DATA_DIR + "/google"
	fonts_url = "https://github.com/google/fonts/archive/master.zip"
	fonts_zipfile = fonts_dir + "/fonts-master"
	fonts_unzipfolder = fonts_dir + "/fonts_unzip"
	ttf_dir = fonts_dir + "/ttf"

	## idempotent download
	if not os.path.isdir(fonts_dir):
		os.mkdir(fonts_dir)
		print("Downloading google fonts..")
		download_url(fonts_url,fonts_zipfile)
		unzip_file(fonts_zipfile,fonts_unzipfolder)

		if not os.path.isdir(ttf_dir):
			os.mkdir(ttf_dir)
			print("Gathering ttf files from google fonts...")
			collect_ttf_files(fonts_unzipfolder,ttf_dir)

def get_1001free_fonts(min_id=20,max_id=28000):

	fonts_dir = DATA_DIR + "/1001free"
	if not os.path.isdir(fonts_dir):
		os.mkdir(fonts_dir)

	fonts_zipdir = fonts_dir + "/zip"
	if not os.path.isdir(fonts_zipdir):
		os.mkdir(fonts_zipdir)

	fonts_ttfdir = fonts_dir + "/ttf"
	if not os.path.isdir(fonts_ttfdir):
		os.mkdir(fonts_ttfdir)

	print("Downloading 1001free fonts..")
	for font_id in range(min_id,max_id):

		fid = str(font_id)
		print("downloading id {id}...".format(id=fid))
		font_url = "https://www.1001freefonts.com/d/{id}/".format(id=fid)
		font_path = fonts_zipdir + "/" + str(fid)

		try:
			download_url(font_url,font_path)
			unzip_file(font_path,fonts_ttfdir + "/" + fid)
		except Exception as e:
			print("error: {e}".format(e=e))

#https://www.dafont.com/alpha.php?lettre=a&page=1&fpp=200

def get_dafont_fonts():

	fonts_dir = DATA_DIR + "/dafont"
	if not os.path.isdir(fonts_dir):
		os.mkdir(fonts_dir)

	fonts_zipdir = fonts_dir + "/zip"
	if not os.path.isdir(fonts_zipdir):
		os.mkdir(fonts_zipdir)

	fonts_ttfdir = fonts_dir + "/ttf"
	if not os.path.isdir(fonts_ttfdir):
		os.mkdir(fonts_ttfdir)

	print("Downloading dafont fonts..")

	my_ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"
	for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
		# parse html of first letter page
		letter_page_url = "https://www.dafont.com/alpha.php?lettre={l}&page=1&fpp=200".format(l=letter.lower())

		raw_html = requests.get(letter_page_url,headers = {"user-agent": my_ua}).text
		page_html = BeautifulSoup(raw_html,"html.parser")
		## find number of letter pages
		letter_pages = page_html.find(class_="noindex").find_all("a")
		page_refs = [page["href"] for page in letter_pages]
		# add first page
		page_refs = ["alpha.php?lettre={l}&page=1&fpp=200".format(l=letter.lower())] + page_refs
		for page in page_refs:
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
				fontname = href.split("=")[-1]
				font_path = fonts_zipdir + "/" + str(fontname)
				try:
					download_url("https:" + href,font_path)
					unzip_file(font_path,fonts_ttfdir + "/" + fontname)
				except Exception as e:
					print("error: {e}".format(e=e))
		
	for font_id in range(min_id,max_id):

		fid = str(font_id)
		print("downloading id {id}...".format(id=fid))
		font_url = "https://www.1001freefonts.com/d/{id}/".format(id=fid)
		font_path = fonts_zipdir + "/" + str(fid)

		try:
			download_url(font_url,font_path)
			unzip_file(font_path,fonts_ttfdir + "/" + fid)
		except Exception as e:
			print("error: {e}".format(e=e))


if __name__=="__main__":

	#get_google_fonts()
	#get_1001free_fonts()
	get_dafont_fonts()
	



