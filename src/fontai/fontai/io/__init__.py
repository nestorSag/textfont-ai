"""This module contains classes that form the data interfaces between ML stages and the outside world (read: storage and other ML stages). Most objects are meant to be used internally by other package modules. Submodules are:

  formats: allowed file format classes; these classes allow to deserialise objects into concrete file formats such as zip and ttf files.

  readers: Batch file readers; each ML stage imposes restrictions on the kind of file that can be read as input, using classes in this module

  scrappers: Scrapper classes to retrieve open source font files using the Ingestion stage.

  storage: Implements ain abstraction that allows to read files from local and remote storage. At the moment, only GCS and public URLs are supported for remote storage.

  writers: Batch file writers
  
"""