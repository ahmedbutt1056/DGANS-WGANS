import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision.utils import make_grid
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="GAN Image Generator", layout="centered")

DCGAN_REPO = "supremeproducts45/dcgan"
WCGAN_REPO = "supremeproducts45/wcgan"

DCGAN_FILE = "dc_g_final.pth"
WCGAN_FILE = "wg_g_final.pth"

z_dim = 100
g_size = 64
img_ch = 3
device = torch.device("cpu")


class Gen(nn.Module):
    def __init__(self, z_dim=100, img_ch=3, g_size=64):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_size * 8, g_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_size * 4, g_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_size * 2, g_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_size, img_ch, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


@st.cache_resource
def load_model(repo_id, file_name):
    path = hf_hub_download(repo_id=repo_id, filename=file_name)
    model = Gen(z_dim=z_dim, img_ch=img_ch, g_size=g_size).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_model(name):
    if name == "DCGAN":
        return load_model(DCGAN_REPO, DCGAN_FILE)
    return load_model(WCGAN_REPO, WCGAN_FILE)


def make_img(name, n):
    model = get_model(name)

    with torch.no_grad():
        z = torch.randn(n, z_dim, 1, 1, device=device)
        fake = model(z).cpu()

    fake = (fake + 1) / 2
    fake = fake.clamp(0, 1)
    grid = make_grid(fake, nrow=min(5, n), padding=4)
    grid = grid.permute(1, 2, 0).numpy()
    img = Image.fromarray((grid * 255).astype("uint8"))
    return img


st.title("GAN Image Generator")
st.write("Generate images using your Hugging Face model weights.")

name = st.selectbox("Choose model", ["DCGAN", "WCGAN"])
n = st.slider("Number of images", 1, 10, 5)

if st.button("Generate"):
    try:
        img = make_img(name, n)
        st.image(img, caption=f"{name} output", use_container_width=True)
    except Exception as e:
        st.error(str(e))
        st.info("If file not found comes, change DCGAN_FILE or WCGAN_FILE at the top of app.py to your exact .pth file names.")
