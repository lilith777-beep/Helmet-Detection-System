# Helmet Detection System using YOLOv8

This project implements a Helmet Detection System using YOLOv8 to automatically detect whether motorcycle riders are wearing helmets. The model is trained on an annotated dataset of riders and uses deep learning based object detection to identify helmets in images and videos.

The system can be used for traffic surveillance, road safety monitoring, and automated violation detection.

---

## Features

- Detects helmets on motorcycle riders
- Uses YOLOv8 object detection model
- Supports image and video inference
- Trained with data augmentation techniques
- Evaluates performance using mAP metrics
- Visualizes detection results

---

## Algorithms and Techniques Used

- YOLOv8 Object Detection Algorithm
- Convolutional Neural Networks (CNN)
- Transfer Learning
- Data Augmentation (Mosaic, MixUp, Horizontal Flip)
- Non-Maximum Suppression (NMS)
- Intersection over Union (IoU)
- Mean Average Precision (mAP) evaluation

---

## Tech Stack

Language
- Python

Frameworks and Libraries
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PIL

---

## Dataset

The model is trained on a helmet detection dataset containing images of motorcycle riders with and without helmets.

Dataset includes:

- Training images
- Validation images
- Bounding box annotations
- Class labels

The dataset is formatted using the YOLO annotation format.

---

## Model Training

The model uses YOLOv8n (nano version) as the base architecture and is trained using the following parameters:

- Epochs: 500  
- Image Size: 736  
- Batch Size: 16  
- Learning Rate: 0.005  

Example training code:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=500,
    imgsz=736,
    batch=16,
    lr0=0.005,
    mosaic=1.0,
    mixup=0.1,
    fliplr=0.5
)
```

---

## Model Evaluation

Model performance is evaluated using object detection metrics such as:

- mAP@0.50
- mAP@0.50–0.95

Example evaluation code:

```python
metrics = model.val()

print(metrics.box.map50)
print(metrics.box.map)
```

---

## Prediction

The trained model can detect helmets in images and videos.

Example image prediction:

```python
model.predict(source="test/images", save=True)
```

Example video detection:

```python
model.predict(source="video1.mp4", save=True)
```

The output will contain bounding boxes around detected helmets.

---

## Project Workflow

1. Install YOLOv8 dependencies
2. Download and prepare the helmet dataset
3. Train the YOLOv8 model
4. Evaluate model performance using validation data
5. Run inference on images and videos
6. Visualize detection results

---

## Project Structure

Helmet-Detection-System

Helmet-Detection-System.ipynb  
data.yaml  
train/  
valid/  
test/  
README.md  

---

## Applications

- Traffic surveillance systems
- Road safety monitoring
- Automated helmet violation detection
- Smart city traffic management

---

## Future Improvements

- Deploy the model using Flask or FastAPI
- Build a real-time helmet detection system using a webcam
- Integrate with traffic cameras
- Improve accuracy using a larger dataset
- Deploy using a Streamlit web interface

---

## Learning Outcomes

- Object detection using YOLOv8
- Training deep learning models using PyTorch
- Dataset annotation and preprocessing
- Model evaluation using mAP metrics
- Running inference on images and videos
