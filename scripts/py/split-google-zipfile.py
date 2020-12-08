import re

def get_fontname(str,ext=".ttf"):
    return str.split("/")[-1].lower().replace(ext,"")

def choose_ext(lst):
  ttfs = len([x for x in lst if ".ttf" in x.lower()])
  otfs = len([x for x in lst if ".otf" in x.lower()])
  if ttfs >= otfs:
    return ".ttf"
  else:
    return ".otf"

def process(inzip,outfolder):
  #logging.info("processing {f}".format(f=gcs_file))
  zip_ = zipfile.ZipFile(inzip)
  #
  files_in_zip = zip_.namelist()
  # choose whether to proces TTFs or OTFs, but not both
  ext = choose_ext(files_in_zip)
  available = sorted([filename for filename in files_in_zip if ext in filename.lower()])
  while len(available) > 0:
    #
    filename = available[0]
    file = filename.split("/")[-1]
    #
    root = re.sub("\\[.+|-.+","",file)
    root_rgx = re.compile(root + "$" + "|" + root + "-" + "|" + root + "\\[")
    matches = [x for x in available if bool(re.match(root_rgx,x.split("/")[-1]))]
    #print("filename: {x}, root: {r}, root_rgx: {rr}".format(x=filename,r=root,rr=root_rgx))
    if len(matches) > 0:
    # create new zipfile 
      with zipfile.ZipFile(outfolder + "/" + root + ".zip","w") as zp:
        for match in np.unique(matches):
          fl = zip_.read(match)
          zp.writestr(match.split("/")[-1],fl)
      #
      available = [x for x in available if not bool(re.match(root_rgx,x.split("/")[-1]))]


inzip = "data/raw/google/zip/fonts-master"
outfolder = "data/raw/google/ttf"
process(inzip,outfolder)