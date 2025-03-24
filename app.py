import os
import pickle
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


# Initialize face detection and recognition models
print("Loading models...")
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')
print("Models loaded successfully!")

# Load face database
face_db = {}
face_db_path = 'face_database.pkl'

def load_face_db():
    global face_db
    if os.path.exists(face_db_path):
        with open(face_db_path, 'rb') as f:
            face_db = pickle.load(f)

def save_face_db():
    with open(face_db_path, 'wb') as f:
        pickle.dump(face_db, f)

# Load database on startup
load_face_db()

def process_image(image_data, name=None):
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Detect faces
        boxes, probs = mtcnn.detect(image)
        if boxes is None:
            return {'success': False, 'message': 'No face detected in the image'}
        
        # Get the first face
        box = boxes[0]
        face = image.crop(box)
        
        # Convert face to tensor
        transform = transforms.ToTensor()
        face_tensor = transform(face).unsqueeze(0)
        
        # Get face embedding
        with torch.no_grad():
            embedding = resnet(face_tensor).squeeze().numpy()
        
        if name:
            # Add to database
            if name not in face_db:
                face_db[name] = []
            face_db[name].append(embedding)
            save_face_db()
            
            # Save face image
            face_path = os.path.join('face_images', f'{name}_{len(face_db[name])}.jpg')
            face.save(face_path)
            
            return {'success': True, 'message': f'Face added to database for {name}'}
        else:
            # Compare with database
            best_match = None
            best_distance = float('inf')
            
            for db_name, embeddings in face_db.items():
                for db_embedding in embeddings:
                    distance = cosine(embedding, db_embedding)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = db_name
            
            if best_match is None:
                return {'success': False, 'message': 'No matching face found in database'}
            
            confidence = 1 - best_distance
            return {
                'success': True,
                'name': best_match,
                'confidence': float(confidence)
            }
            
    except Exception as e:
        return {'success': False, 'message': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_face', methods=['POST'])
def add_face():
    data = request.json
    image_data = data.get('image')
    name = data.get('name')
    
    if not image_data or not name:
        return jsonify({'success': False, 'message': 'Missing image or name'})
    
    result = process_image(image_data, name)
    return jsonify(result)

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'Missing image'})
    
    result = process_image(image_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)