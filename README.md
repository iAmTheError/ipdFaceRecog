# FaceNet Implementation

This project implements face recognition using the FaceNet algorithm with PyTorch.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your test image in the project directory
2. Update the `image_path` variable in `facenet_demo.py` with your image path
3. Run the script:
```bash
python facenet_demo.py
```

The script will:
- Detect faces in the input image
- Generate face embeddings using FaceNet
- Display the image with detected faces marked with red boxes
- Print the number of faces detected and the shape of the embeddings

## Features

- Face detection using MTCNN
- Face embedding generation using FaceNet (InceptionResnetV1)
- Visualization of detected faces
- Support for both CPU and GPU (CUDA)

## Requirements

- Python 3.7+
- PyTorch
- facenet-pytorch
- Other dependencies listed in requirements.txt #   i p d F a c e R e c o g  
 