from pydantic import BaseModel

class InMemoryFile(BaseModel):
  # wrapper that holds the bytestreams of unzipped in-memory ttf/otf files
  filename: str
  content: bytes

class LabeledExample(BaseModel):

  x: object
  y: str
  metadata: str

  def __iter__(self):
    return(iter(x,y,metadata))


class KeyValuePair(BaseModel):

  key: str
  value: object

  def __iter__(self):
    return(iter(key,value))