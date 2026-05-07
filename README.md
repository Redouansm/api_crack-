````md id="jlwm77"
# FastAPI Image Segmentation API

This project is a FastAPI-based AI segmentation API using a trained DeepLabV3+ ResNet101 model.

The API:
- receives an image,
- generates a segmentation mask,
- creates a segmented image overlay,
- returns the generated images.

---

# Project Structure

```bash
project/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   └── best_model_resnet101.pth
│
├── outputs/
````

---

# Technologies Used

* FastAPI
* PyTorch
* segmentation-models-pytorch
* DeepLabV3+
* ResNet101

---

# Installation

## 1. Clone or Download Project

Download the GitHub project.

---

## 2. Install Python Dependencies

Open terminal inside the project folder and run:

```bash
pip install -r requirements.txt
```

---

# Download Trained Model

Download the trained model file:

```text
best_model_resnet101.pth
```
####  https://drive.google.com/file/d/1qDhMkpwWkczMnn4pwNUOLVRlXcnmj17H/view?usp=sharing

Place it inside:

```bash
model/best_model_resnet101.pth
```

---

# Run FastAPI Server

Run:

```bash
uvicorn main:app --reload
```

Server starts on:

```text
http://127.0.0.1:8000
```

---

# Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

This automatically provides a testing interface for the API.

---

# API Endpoint

## POST `/predict`

Upload an image for segmentation.

### Input

Image file.

### Output

JSON response:

```json
{
  "original_image": "...",
  "mask_image": "...",
  "segmented_image": "..."
}
```

---

# Example Workflow

```text
Frontend / Spring Boot
        ↓
    FastAPI API
        ↓
 DeepLabV3+ Model
        ↓
 Generate Mask
        ↓
 Return Results
```

---

# Generated Files

If uploaded image is:

```text
leaf.jpg
```

The API generates:

```text
leaf_original.jpg
leaf_mask.png
leaf_segmented.jpg
```

inside:

```bash
outputs/
```

---

# Spring Boot Integration

Spring Boot sends images to:

```text
http://127.0.0.1:8000/predict
```

FastAPI returns:

* original image,
* mask image,
* segmented image.

---

# Notes

* The `outputs/` folder is ignored by Git.
* The trained `.pth` model is not stored on GitHub.
* The model must be manually added inside the `model/` folder.

---



```
```
