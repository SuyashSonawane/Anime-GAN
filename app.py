import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import random
import base64
import streamlit as st

t1 = 0.5
t2 = 0.1
t3 = 0.02
torch.manual_seed(2)

with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.write(
    """# Anime Gen
Anime Gen generates a fake anime face using a GAN which learned from a dataset of 1k anime face drawings.
You can also control the output by changing the values of top 3 features learned by the Neural Network
"""
)

model = torch.load("G-1.ckpt").to("cpu")
model.eval()

if st.sidebar.button("Generate New Face"):
    torch.manual_seed(random.randint(0, 500))

st.sidebar.write("## Top 3 Features affecting the images")

t1 = st.sidebar.slider("T1", -500, 500, 1) * 0.01
t2 = st.sidebar.slider("T2", -500, 500, 1) * 0.01
t3 = st.sidebar.slider("T3", -50, 500, 1) * 0.01

st.sidebar.markdown(
    "Get the Source Code [here](https://github.com/SuyashSonawane?tab=repositories)",
    unsafe_allow_html=True,
)


def generateFace(t1, t2, t3):
    latent = torch.randn(1, 128, 1, 1, device="cpu")
    latent[0][0][0] = t1
    latent[0][1][0] = t2
    latent[0][2][0] = t3
    fake_images = model(latent)
    print(fake_images.shape)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
    return fig


st.set_option("deprecation.showPyplotGlobalUse", False)
st.pyplot(generateFace(t1, t2, t3))


st.markdown("### This project is just for my learning purpose")
st.markdown(" ### Below is the animation of how the GAN got better steadily")
st.image("ezgif-5-f45f6235c08b.gif")
