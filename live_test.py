import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import load_model

actions = np.array(['hello', 'thanks', 'iloveyou', 'please', 'sorry', 'ok', 'welcome', 'help', 'learn'])
threshold = 0.5

model = load_model('action.h5')

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(128, 128, 0), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 126, 255), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(100, 200, 255), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(0, 126, 255), thickness=4, circle_radius=5), 
                             mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(0, 126, 255), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                             ) 

sequence = []
current_word = ""  # Stores only the current word
predictions = []

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        
        # Reset display if no hands detected
        hands_detected = results.left_hand_landmarks or results.right_hand_landmarks
        if not hands_detected:
            current_word = ""
        
        # Prediction logic (only when hands are detected)
        if hands_detected:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep last 30 frames
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                current_prediction = np.argmax(res)
                predictions.append(current_prediction)
                
                # Update word only when we have 10 consistent predictions
                if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == current_prediction:
                    if res[current_prediction] > threshold:
                        new_word = actions[current_prediction]
                        
                        # Only update if it's a different word
                        if new_word != current_word:
                            current_word = new_word
        
        # Display current word on the right side (only if word exists)
        if current_word:
            # Position parameters
            text_size = cv2.getTextSize(current_word, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            margin = 20
            box_x = image.shape[1] - text_size[0] - margin*100
            box_y = margin*20
            
            # Text background
            cv2.rectangle(image, 
                         (image.shape[1] - text_size[0] - margin*2, margin),
                         (image.shape[1], margin + text_size[1] + margin),
                         (245, 117, 16), -1)
            
            # Word text
            cv2.putText(image, current_word, 
                       (image.shape[1] - text_size[0] - margin, margin + text_size[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        cv2.imshow('Action Recognition', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()