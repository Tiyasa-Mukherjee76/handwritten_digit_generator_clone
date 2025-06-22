# Handwritten Digit Image Generator (MNIST)

This project is a simple web application that allows users to generate handwritten digit images (0–9) using a custom-trained neural network. The model is trained from scratch on the **MNIST dataset** without any pretrained weights and deployed using **Streamlit**.

---

## 🔧 Features

* Choose a digit (0 to 9)
* Generate 5 different handwritten-style images of the selected digit
* Model trained from scratch using PyTorch (no pretrained weights)
* Clean and interactive user interface with Streamlit
* Deployable to Streamlit Cloud or any public server

---

## 🧠 Model Architecture

* Fully connected generator neural network
* Input: latent noise vector + digit class embedding
* Output: 28x28 grayscale image
* Trained on MNIST (60,000 training samples)

---

## 🗂 Project Structure

```
📦digit-generator-app/
 ┣ 📄 app.py              # Streamlit app
 ┣ 📄 model.py            # Generator model architecture
 ┣ 📄 digit_generator.pth # Trained model weights
 ┣ 📄 train_model.ipynb   # Jupyter notebook to train the model from scratch
 ┗ 📄 README.md
```

---

## 🚀 How to Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/digit-generator-app.git
   cd digit-generator-app
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 📦 Deployment

* Push this repo to GitHub.
* Visit [Streamlit Cloud](https://share.streamlit.io).
* Connect your repo and deploy in seconds.

---

## 📝 Notes

* This project complies with academic restrictions: **no pretrained models used**.
* Designed and implemented as part of a hands-on AI examination prompt.
