# ✍️ Handwritten Digit Recognition System

A deep learning application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The system features an interactive Streamlit web app where users can draw a digit or upload an image for real-time prediction.

## 🚀 Live Demo

🔗 https://handwrittendigitrecognizer-nv4rjiuds8eev4dciwjqky.streamlit.app/

## 🚀 Features

- CNN model trained on the MNIST dataset for high-accuracy digit classification
- Interactive web app to draw digits directly on a canvas
- Option to upload custom digit images for prediction
- Real-time prediction with confidence scores
- User-friendly and responsive Streamlit interface
- Instant visualization of prediction results

## 🛠️ Tech Stack

- **Python** – Core programming language
- **TensorFlow / Keras** – Model building and training
- **NumPy** – Numerical operations and array processing
- **Streamlit** – Interactive web-based user interface
- **Matplotlib** – Data visualization and training metrics
- **Pillow (PIL)** – Image processing and handling


## 📂 Project Structure

```text
handwritten-digit-recognition/
├── assets/
│   ├── app_screenshot.png
│   ├── drawing_demo.png
│   └── prediction_results.png
├── model/                   # Saved trained model files
├── train_model.py           # Script to train the CNN model on MNIST
├── app.py                   # Streamlit application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/Mac**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

This will launch the web interface in your browser, where you can:

- Draw a digit on the interactive canvas.
- Upload an image containing a handwritten digit.
- Get an instant prediction along with confidence scores.

## 🧠 Model Training

To train the CNN model on the MNIST dataset from scratch:

```bash
python train_model.py
```

The trained model will be saved in the `model/` directory and automatically loaded by the Streamlit application.

## 📦 Requirements

```text
streamlit==1.58.0
tensorflow==2.19.0
numpy==2.1.3
matplotlib==3.10.5
pillow==11.3.0
```

## 📊 Results

The CNN model achieves high accuracy (typically above **98%**) on the MNIST test dataset, making it reliable for recognizing a wide variety of handwriting styles.

The system can successfully:

- Recognize digits from 0–9 with high accuracy.
- Predict digits drawn on the canvas in real time.
- Classify uploaded handwritten digit images efficiently.
- Demonstrate practical applications of deep learning and computer vision.
