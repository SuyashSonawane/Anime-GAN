import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from flask import Response, request
import random
from flask import Flask
from flask import render_template
import io
app = Flask(__name__)

t1 = 0.5
t2 = 0.1
t3 = 0.02

model = torch.load("G-1.ckpt").to("cpu")
model.eval()


def generateFace(t1, t2, t3):
    latent = torch.randn(1, 128, 1, 1, device="cpu")
    latent[0][0][0] = t1
    latent[0][1][0] = t2
    latent[0][2][0] = t3
    fake_images = model(latent)
    print(fake_images.shape)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
    return fig


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/plot.png')
def plot_png():
    t1 = float(request.args.get("t1"))
    t2 = float(request.args.get("t2"))
    t3 = float(request.args.get("t3"))
    fig = generateFace(t1, t2, t3)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)