# Gender Detection 🎯

A simple AI-powered system that detects gender from images using YOLO and a custom-trained model.

## 📁 Project Structure

The project consists of the following Python scripts:

- `downloadDataset.py` – Downloads a gender classification dataset from Roboflow.
- `createModel.py` – Creates and trains a YOLO-based AI model for gender detection.
- `detect.py` – Detects gender from:
  - a single image
  - a folder of images
  - a live webcam feed

> 🔧 To switch between detection modes, edit the last few lines in `detect.py` – comment out unused options and uncomment the one you want to use.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PiotrSaw/GenderDetection.git
cd GenderDetection
```

### 2. Install dependencies

Make sure you have Python 3.x installed. Then install required libraries:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install opencv-python numpy ultralytics
```

### 3. Download the dataset

```bash
python downloadDataset.py
```

This will fetch the dataset from Roboflow using an API key (you may need to sign up on Roboflow and get your own key).

### 4. Train the model

```bash
python createModel.py
```

This trains a new YOLO model for gender detection based on the downloaded dataset.

### 5. Run gender detection

```bash
python detect.py
```

Choose your detection method inside the script:
- `detectFromImage()`
- `detectFromFolder()`
- `detectFromWebcam()`

---

## 🧠 Model Info

The project uses YOLO (You Only Look Once) object detection framework for real-time performance and high accuracy. Training and inference are handled through the `ultralytics` library.

---

## 👤 Author

**Piotr S.** – [GitHub Profile](https://github.com/PiotrSaw)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
