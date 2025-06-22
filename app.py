import torch
import streamlit as st
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import Generator  # Make sure model.py defines Generator

# Load the trained generator model
latent_dim = 100
num_classes = 10
model = Generator(latent_dim, num_classes)
model.load_state_dict(torch.load("digit_generator.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate Images"):
    noise = torch.randn(5, latent_dim)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        images = model(noise, labels)

    grid = make_grid(images, nrow=5, normalize=True, pad_value=1)
    plt.figure(figsize=(10, 2))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    st.pyplot(plt)  # ✅ FIXED: removed the extra dot
    plt.clf()       # Optional: clear figure after render
