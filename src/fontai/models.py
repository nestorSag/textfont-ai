import tensorflow as tf


def get_layer_instance(layer_name):
  ## returns the layer class based on its name; looks in the tf and local modules
  try:
    layer = getattr(tf.keras.layers,layer_name)
    return layer
  except AttributeError as e:
    try:
      layer = getattr(__name__,layer_name)
      return layer
    except Exception as e:
      print(f"error instantiating layer class: {e}")

class StackedNetwork(tf.keras.Model):
  """ Network formed by stacking layers

  Parameters:

  layers (`dict`): dict with a "layers" key, whose value is a list of dicts with two keys: "class" and "kwargs".
  Class is the name of the class from the tf.keras.layers package

  """
  def __init__(self,layers):
    super(StackedNetwork,self).__init__()
    layer_number = 0
    for layer in layers["layers"]:
      #layer_class = getattr(tf.keras.layers,layer["class"])
      layer_class = get_layer_instance(layer["class"])
      setattr(self,"layer" + str(layer_number),layer_class(layer["kwargs"]))
      layer_number += 1
    self.n_layers = layer_number

  def call(self,inputs):
    for k in range(self.n_layers):
      x = getattr(self,"layer" + str(k))(x)
    return x

class ScatterGatherConvLayer(tf.keras.layer.Layer):
  """ This layer passes its input through multiple separate layers and then concatenate their output in a single tensor
      #with same dimensions as input
  Parameters:

  layers_info (`dict`): dict with a "submodules" key, whose value is a list of dicts that are inputs for `StackedNetwork` instances
  """

  def __init__(self,layers):
    super(ScatterGatherConvLayer,self).__init__()
    module_number = 0
    for module in layers["submodules"]:
      #layer_class = getattr(tf.keras.layers,layer["class"])
      setattr(self,"module" + str(module_number),StackedNetwork(module))
      module_number += 1
    self.module_number = module_number

  def call(self,inputs):
    return tf.concat(getattr(self,"module" + str(i))(inputs) for i in range(self.module_number), axis=3)

# {"submodules":[{"layers":[...]},{"layers":[...]}]}

