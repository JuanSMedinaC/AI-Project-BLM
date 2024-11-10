import cv2
import mediapipe as mp
import csv
from datetime import datetime
import os
from time import sleep

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Generate a timestamp with date and hour (with seconds)
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print(f"Timestamp: {timestamp}")

output_csv = 'data/' + timestamp + '.csv'

# Open the webcam feed (change '0' to the index of the camera you want to use)
cap = cv2.VideoCapture(0)

def write_landmarks_to_csv(landmarks, frame_number, timestamp, csv_data):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z}), timestamp: {timestamp}")
        csv_data.append([frame_number, timestamp, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
    print("\n")

frame_number = 0
csv_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        sleep(1/30)
        # Add the landmark coordinates to the list and print them
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, timestamp, csv_data)

    # Display the frame with landmarks
    cv2.imshow('MediaPipe Pose - Real Time', frame)

    # Increment the frame number
    frame_number += 1

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()

# Write collected data to CSV file after exiting
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Timestamp", "Landmark", "X", "Y", "Z"])
    writer.writerows(csv_data)

print("Data saved to CSV file.")
