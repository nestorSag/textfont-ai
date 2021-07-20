import io
import base64
import typing as t

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

embedding_dim = 10
font_size = 26
model = tf.keras.models.load_model("models/font-style-generative/decoder")

app.layout = html.Div(children=[
  html.Div(children=[html.Button('Random', id='button')] +
    [dcc.Slider(min=-3,max=3,step=0.1, value=0, id=f"slider-{k}") for k in range(embedding_dim)],
    style = {'display': 'inline-block', 'width': '25%'}),
  html.Img(id="font_figure")
  ])



def fig_to_str(in_fig, close_all=True, **save_args):
  # type: (plt.Figure) -> str
  """
  Save a figure as a URI
  :param in_fig:
  :return:
  """
  out_img = io.BytesIO()
  in_fig.savefig(out_img, format='png', **save_args)
  if close_all:
      in_fig.clf()
      plt.close('all')
  out_img.seek(0)  # rewind file
  encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
  return f"data:image/png;base64,{encoded}"


def generate_font(style):

  # sample one hot encoded labels
  labels = []
  for k in range(font_size):
    onehot = np.zeros((1,font_size), dtype=np.float32)
    onehot[0,k] = 1.0
    labels.append(np.concatenate([style,onehot],axis=-1))
  
  fully_encoded = np.array(labels, dtype=np.float32)

  chars = model.predict(fully_encoded)
  return chars

def plot_font(imgs: t.Union[tf.Tensor, np.ndarray], n_cols = 7) -> None:
  """Utility function to plot a sequence of characters and save it in a given location as a single tiled figure.
  
  Args:
      imgs (t.Union[tf.Tensor, np.ndarray]): 4-dimensional array of images
      output_file (str): output file
      n_cols (int, optional): number of columns in output figure
  """
  # plot multiple images
  n_imgs, height, width, c = imgs.shape
  n_rows = int(np.ceil(n_imgs/n_cols))

  fig, axs = plt.subplots(n_rows, n_cols)
  for i in range(n_imgs):
    x = imgs[i]
    if isinstance(x, np.ndarray):
      x_ = x
    else:
      x_ = x.numpy()
    np_x = (255 * x_).astype(np.uint8).reshape((height,width))
    axs[int(i/n_cols), i%n_cols].imshow(np_x, cmap="Greys")

  for k in range(n_rows*n_cols):
    axs[int(k/n_cols), k%n_cols].axis("off")
  
  return fig


@app.callback([
  Output(f"slider-{k}","value") for k in range(embedding_dim)],
  Input("button","n_clicks"))
def update_random(n_clicks):
  return list(np.clip(np.random.normal(size=embedding_dim), a_min=-3, a_max=3))

@app.callback(
    Output('font_figure', 'src'),
    [Input(f"slider-{k}", 'value') for k in range(embedding_dim)])
def update(*args):
  style = np.array(args).reshape((1,-1))
  font = generate_font(style)
  img = plot_font(font)
  return fig_to_str(img)

if __name__ == '__main__':
  app.run_server(debug=True)