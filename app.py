import os
import json
import base64
import numpy as np
from PIL import Image
import io
from flask import Flask, render_template, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import cv2
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Create necessary directories
os.makedirs('face_images', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

# Initialize face detection and recognition models
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

# Load face database
face_db = {}
face_db_path = 'face_db.json'

def load_face_db():
    global face_db
    if os.path.exists(face_db_path):
        with open(face_db_path, 'r') as f:
            face_db = json.load(f)

def save_face_db():
    with open(face_db_path, 'w') as f:
        json.dump(face_db, f)

# Load database on startup
load_face_db()

// ... existing code ...