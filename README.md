# 🩺 Medical Image Classification using Computer Vision

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Torchvision](https://img.shields.io/badge/Torchvision-Computer%20Vision-orange)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-green)
![Grad-CAM](https://img.shields.io/badge/Explainable%20AI-GradCAM-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A desktop application for automatic classification of chest X-ray images using **Deep Learning**.

The application is based on **ResNet18** with **Transfer Learning** and provides visual explanations of predictions using **Grad-CAM**. It helps classify chest X-ray images into **Normal** and **Pneumonia** classes while highlighting the image regions that most influenced the model's decision.

---

# 🔗 Repository

**GitHub Repository**

https://github.com/Vitalik1800/medicalsystem

> Replace the repository name if you choose a different one.

---

# 🚀 Features

## 🖼 Medical Image Processing

* Load chest X-ray images
* Image preprocessing
* Resize and normalization
* Automatic image classification
* Prediction confidence score
* Grad-CAM heatmap visualization

---

## 🧠 Deep Learning

* ResNet18 architecture
* Transfer Learning
* Fine-tuning on medical images
* GPU support (CUDA if available)

---

## 🖥 Desktop Application

* User-friendly Tkinter interface
* Image preview
* One-click prediction
* Grad-CAM visualization
* Classification result display

---

# 🛠 Tech Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Programming Language    |
| PyTorch      | Deep Learning Framework |
| Torchvision  | Image Processing        |
| Pillow (PIL) | Image Loading           |
| Tkinter      | Desktop GUI             |
| NumPy        | Numerical Computing     |
| OpenCV       | Image Processing        |
| Grad-CAM     | Explainable AI          |

---

# 📂 Project Structure

```text
MedicalImageClassifier/
│
├── dataset/
│
├── screenshots/
│
├── image_loader.py
├── main.py
├── model.py
├── processor.py
├── train.py
├── ui.py
│
└── README.md
```

---

# 📸 Screenshots

All application screenshots are stored in the **screenshots** folder.

```
screenshots/
```

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Vitalik1800/medicalsystem.git
```

## 2. Open the project directory

```bash
cd medicalsystem
```

## 3. Run the application

```bash
python main.py
```

---

# 📊 Model Information

| Parameter         | Value              |
| ----------------- | ------------------ |
| Model             | ResNet18           |
| Framework         | PyTorch            |
| Classes           | Normal / Pneumonia |
| Transfer Learning | Yes                |
| Explainability    | Grad-CAM           |
| Optimizer         | Adam               |
| Scheduler         | ReduceLROnPlateau  |
| Epochs            | 20                 |
| Batch Size        | 16                 |

---

# 📈 Model Performance

| Metric                |    Value |
| --------------------- | -------: |
| Accuracy              |  **80%** |
| Precision (Normal)    |  **99%** |
| Recall (Normal)       |  **47%** |
| F1-score (Normal)     |  **63%** |
| Precision (Pneumonia) |  **76%** |
| Recall (Pneumonia)    | **100%** |
| F1-score (Pneumonia)  |  **86%** |

---

# 🎯 Learning Objectives

This project demonstrates practical experience with:

* Computer Vision
* Deep Learning
* Medical Image Classification
* Transfer Learning
* Explainable Artificial Intelligence (XAI)
* PyTorch
* Desktop Application Development
* Neural Networks
* Image Processing

---

# 💡 Future Improvements

Possible future enhancements include:

* Support for DICOM images
* Multi-class disease classification
* Larger medical datasets
* Improved model accuracy
* Web application version
* REST API integration
* Medical Information System integration
* Better Grad-CAM visualization
* Real-time inference

---

# 👨‍💻 Author

**Vitalii Semchyshyn**

Junior Python Developer

**GitHub**

https://github.com/Vitalik1800

**LinkedIn**

https://linkedin.com/in/віталій-семчишин-1557292a8

---

# ⭐ Project Highlights

* 🧠 Deep Learning
* 🩺 Medical AI
* 🔥 Grad-CAM Visualization
* 📷 Medical Image Classification
* ⚡ PyTorch
* 💻 Desktop Application
* 🖥 Tkinter GUI
* 📊 Transfer Learning
* 🎯 Explainable AI
* 🚀 Portfolio Project

---

⭐ If you found this project interesting, feel free to **star the repository**.
