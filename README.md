# ✍️ Handwritten Digit Recognition System

A deep learning application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The system features an interactive Streamlit web app where users can draw a digit or upload an image for real-time prediction.

## 🚀 Features

- CNN model trained on the MNIST dataset for high-accuracy digit classification
- Interactive web app to draw digits directly on a canvas
- Option to upload custom digit images for prediction
- Real-time prediction with confidence scores

## 🛠️ Tech Stack

- **Python** – Core programming language
- **TensorFlow / Keras** – Model building and training
- **Streamlit** – Interactive web-based user interface
- **Matplotlib** – Data visualization and training metrics

## 📂 Project Structure

```
handwritten-digit-recognition/
├── model/                   # Saved trained model files
├── train_model.py           # Script to train the CNN model on MNIST
├── app.py                   # Streamlit application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## ⚙️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

This will launch the web interface in your browser, where you can draw a digit on the canvas or upload an image to get an instant prediction.

## 🧠 Model Training

To train the CNN model on the MNIST dataset from scratch:
```bash
python train_model.py
```

The trained model will be saved in the `model/` directory for use in the application.

## 📊 Results

The CNN model achieves high accuracy (typically above 98%) on the MNIST test dataset, making it reliable for recognizing a wide variety of handwriting styles.
