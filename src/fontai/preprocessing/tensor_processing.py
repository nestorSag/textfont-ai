class InputDataHandler(object):

  def __init__(self,padding=2,pixel_threshold=100,charset="all",img_dim=(64,64)):

    self.img_dim = img_dim

    self.record_spec = {
      'char': tf.io.FixedLenFeature([], tf.string),
      'filename': tf.io.FixedLenFeature([], tf.string),
      'img': tf.io.FixedLenFeature([], tf.string),
    }

    if charset != "all":
      if charset == "lowercase":
        self.classes = string.ascii_letters[0:26]
      elif charset == "uppercase":
        self.classes = string.ascii_letters[26::]
      elif charset == "numbers":
        self.classes = string.digits
      else:
        raise Exception("Only 'lowercase', 'uppercase', 'numbers' or 'all' are allowed as options for charset")
    else:
      self.classes = string.ascii_letters + string.digits

    self.tf_classes = tf.convert_to_tensor(list(self.classes))
    self.num_classes = len(self.classes)
    self.padding = padding
    self.pixel_threshold = pixel_threshold
    self.charset = charset

  def parse_tf_objects(self,serialized):
    return tf.io.parse_single_example(serialized,self.record_spec)

  def filter_by_char(self,parsed):
    return tf.reduce_any(self.tf_classes == parsed["char"])

  def process_tf_objects(self,parsed):
    img = tf.image.decode_png(parsed["img"])
    img = tf.image.resize_with_crop_or_pad(img,target_height=self.img_dim[0]+2*self.padding,target_width=self.img_dim[1]+2*self.padding)
    img = tf.cast(img,dtype=tf.float32)
    y = tf.cast(tf.where(self.tf_classes == parsed["char"]),dtype=tf.int32)
    label = tf.reshape(tf.one_hot(indices=y,depth=self.num_classes),(self.num_classes,))#.reshape((num_classes,))
    return img, label, parsed["filename"]

  # def filter_sparse_images(self,img,label):
  #   return tf.math.count_nonzero(img) > self.pixel_threshold

  def filter_sparse_images(self,img,label,filename):
    return tf.math.count_nonzero(img) > self.pixel_threshold

  @classmethod
  def remove_filename(self,img,label,filename):
    return img, label

  def get_dataset(self,folders,include_fontname=False):

    if not isinstance(folders,list):
      folders = [folders]

    def standardise_folder_name(folder):
      if folder[-1] != "/":
        folder = folder + "/"

      return folder

    folders = [standardise_folder_name(folder) for folder in folders]

    files = [folder + file for folder in folders for file in os.listdir(folder)]

    dataset = tf.data.TFRecordDataset(filenames=files)\
      .map(self.parse_tf_objects)\
      .filter(self.filter_by_char)\
      .map(self.process_tf_objects)\
      .filter(self.filter_sparse_images)

    return dataset

  def scramble_dataset(self,dataset,batch_size=32):
    dataset = dataset\
      .shuffle(buffer_size=2*batch_size)\
      .repeat()\
      .batch(batch_size)

    return dataset

  def get_training_dataset(self,folders,batch_size=32):
    return self.scramble_dataset(self.get_dataset(folders).map(InputDataHandler.remove_filename),batch_size=batch_size)

  def get_evaluation_dataset(self,folders,batch_size=None):
    dataset = self.get_dataset(folders).map(InputDataHandler.remove_filename)
    
    if batch_size is not None:
      return dataset.batch(batch_size)
    else:
      return dataset

  def supervised_filter(self,model):
    def filter_func(imgs,labels,filenames):
      # filters a batch using a trained model
      pred = model(imgs)
      condition = tf.argmax(pred,axis=-1) == tf.argmax(labels,axis=-1)
      return imgs[condition], labels[condition], filenames[condition]

    return filter_func

  def expand_labels(self,img,label,filename):

    lowercase_labels = tf.zeros((26,)) if self.charset != "lowercase" else label
    uppercase_labels = tf.zeros((26,)) if self.charset != "uppercase" else label
    number_labels = tf.zeros((10,)) if self.charset != "numbers" else label

    return img, tf.concat([lowercase_labels,uppercase_labels,number_labels],axis=0), filename

  @classmethod
  def read_gen_model_data(self,path,padding=0,img_dim=(64,64)):
    elemspec= (tf.TensorSpec(shape=(img_dim[0]+2*padding,img_dim[1]+2*padding,1), dtype=tf.float32, name=None), tf.TensorSpec(shape=(62,), dtype=tf.float32, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None))

    return tf.data.experimental.load(path,element_spec = elemspec,compression="GZIP")

  @classmethod
  def filter_by_char(cls,char):
    idx = list(string.ascii_letters + string.digits).index(char)
    if char == "all":
      def f():
        return True
    else:
      def f(img,label,*args):
        return tf.argmax(label) == idx

    return f
