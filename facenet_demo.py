from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import numpy as np
from PIL import Image
import time
from face_recognition import FaceRecognition
import os
import subprocess

def load_model():
    # Initialize face detection pipeline
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize FaceNet model
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet = resnet.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return mtcnn, resnet

def view_database(recognition):
    """Display all names in the database"""
    if not recognition.face_database:
        print("\nDatabase is empty! No faces have been added yet.")
        return
    
    print("\nPeople in the database:")
    print("-" * 30)
    for name in recognition.face_database.keys():
        num_samples = len(recognition.face_database[name])
        print(f"{name}: {num_samples} face sample{'s' if num_samples > 1 else ''}")
    print("-" * 30)

def recognize_face_from_image(image_path, mtcnn, resnet, recognition):
    """Recognize face from an image file"""
    # Load and process image
    img = Image.open(image_path)
    boxes, probs = mtcnn.detect(img)
    
    if boxes is not None:
        faces = mtcnn(img)
        if faces is not None:
            embeddings = resnet(faces).detach().cpu().numpy()
            
            # Recognize each face
            for i, embedding in enumerate(embeddings):
                name, confidence = recognition.recognize_face(embedding)
                print(f"Face {i+1}: {name} (Confidence: {1-confidence:.2f})")
    else:
        print("No faces detected in the image")

def recognize_from_camera(mtcnn, resnet, recognition):
    """Recognize faces from live camera feed"""
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Detect faces
            boxes, probs = mtcnn.detect(frame_pil)
            
            if boxes is not None:
                # Draw boxes around faces
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract face embeddings
                faces = mtcnn(frame_pil)
                if faces is not None:
                    embeddings = resnet(faces).detach().cpu().numpy()
                    
                    # Recognize each face
                    for i, embedding in enumerate(embeddings):
                        name, confidence = recognition.recognize_face(embedding)
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        # Display recognition info
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.putText(frame, f"Name: {name}", (x1, y1-40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Confidence: {1-confidence:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show the frame
            cv2.imshow('Face Recognition (Press Q to quit)', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Load models
    print("Loading models...")
    mtcnn, resnet = load_model()
    recognition = FaceRecognition()
    print("Models loaded successfully!")
    
    while True:
        print("\nFace Recognition System")
        print("1. Recognize face from image")
        print("2. Recognize face from camera")
        print("3. Add new face to database")
        print("4. View database")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            image_path = input("Enter path to image file: ")
            if os.path.exists(image_path):
                recognize_face_from_image(image_path, mtcnn, resnet, recognition)
            else:
                print("Image file not found!")
                
        elif choice == '2':
            recognize_from_camera(mtcnn, resnet, recognition)
            
        elif choice == '3':
            print("\nLaunching face addition program...")
            subprocess.run(['python', 'add_face.py'])
            
        elif choice == '4':
            view_database(recognition)
            
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main() 