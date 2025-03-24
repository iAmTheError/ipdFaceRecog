from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import numpy as np
from PIL import Image
from face_recognition import FaceRecognition
import os
from datetime import datetime

def load_model():
    # Initialize face detection pipeline
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize FaceNet model
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet = resnet.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return mtcnn, resnet

def save_face_image(frame, boxes, name):
    """Save the entire frame as a JPG file"""
    # Create faces directory if it doesn't exist
    if not os.path.exists('face_images'):
        os.makedirs('face_images')
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"face_images/{name}_{timestamp}.jpg"
    
    # Save the entire frame
    cv2.imwrite(filename, frame)
    print(f"Image saved as: {filename}")
    return filename

def capture_face(mtcnn, resnet):
    """Capture a single frame and detect face"""
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
                    if len(embeddings) > 0:
                        return embeddings[0], frame, boxes  # Return embedding, frame, and boxes
            
            # Show the frame
            cv2.imshow('Press SPACE to capture, ESC to cancel', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return None, None, None
            elif key == 32:  # SPACE
                if boxes is not None and len(boxes) > 0:
                    return embeddings[0], frame, boxes
                else:
                    print("No face detected! Try again.")
    
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
        print("\nAdd Face to Database")
        print("1. Capture face from camera")
        print("2. Add face from image file")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            name = input("Enter name for the face: ")
            print("\nPosition your face in front of the camera")
            print("Press SPACE to capture, ESC to cancel")
            
            embedding, frame, boxes = capture_face(mtcnn, resnet)
            if embedding is not None:
                # Save the face image
                save_face_image(frame, boxes, name)
                # Add to database
                recognition.add_face(name, embedding)
                print(f"Successfully added face for {name}")
            else:
                print("Face capture cancelled")
                
        elif choice == '2':
            image_path = input("Enter path to image file: ")
            if os.path.exists(image_path):
                # Load and process image
                img = Image.open(image_path)
                boxes, probs = mtcnn.detect(img)
                
                if boxes is not None:
                    faces = mtcnn(img)
                    if faces is not None:
                        embeddings = resnet(faces).detach().cpu().numpy()
                        if len(embeddings) > 0:
                            name = input("Enter name for the face: ")
                            # Convert PIL image to OpenCV format for saving
                            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            # Save the face image
                            save_face_image(frame, boxes, name)
                            # Add to database
                            recognition.add_face(name, embeddings[0])
                            print(f"Successfully added face for {name}")
                        else:
                            print("No face detected in the image!")
                else:
                    print("No face detected in the image!")
            else:
                print("Image file not found!")
                
        elif choice == '3':
            print("Returning to main menu...")
            break
            
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main() 