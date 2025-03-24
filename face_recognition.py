import numpy as np
import os
import pickle
from scipy.spatial.distance import cosine
import cv2

class FaceRecognition:
    def __init__(self, database_path='face_database.pkl'):
        self.database_path = database_path
        self.face_database = self.load_database()
        
    def load_database(self):
        """Load the face database from file or create new if doesn't exist"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_database(self):
        """Save the face database to file"""
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.face_database, f)
    
    def add_face(self, name, embedding):
        """Add a new face to the database"""
        if name not in self.face_database:
            self.face_database[name] = []
        self.face_database[name].append(embedding)
        self.save_database()
    
    def recognize_face(self, embedding, threshold=0.6):
        """Recognize a face by comparing its embedding with the database"""
        best_match = None
        best_distance = float('inf')
        
        for name, embeddings in self.face_database.items():
            # Compare with all embeddings of this person
            for stored_embedding in embeddings:
                distance = cosine(embedding, stored_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
        
        if best_distance < threshold:
            return best_match, best_distance
        return "Unknown", best_distance
    
    def display_recognition_info(self, frame, name, confidence, position):
        """Display recognition information on the frame"""
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
        else:
            color = (0, 255, 0)  # Green for recognized
            
        cv2.putText(frame, f"Name: {name}", (position[0], position[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Confidence: {1-confidence:.2f}", (position[0], position[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    # Example usage
    recognition = FaceRecognition()
    
    # Example of adding a face (you would get the embedding from your detection code)
    # sample_embedding = np.random.rand(512)  # Replace with actual embedding
    # recognition.add_face("John", sample_embedding)
    
    # Example of recognizing a face
    # name, confidence = recognition.recognize_face(sample_embedding)
    # print(f"Recognized as: {name} with confidence: {1-confidence:.2f}")

if __name__ == "__main__":
    main() 