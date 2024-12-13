import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier

def speak(str1):
    """Function to speak the text using SAPI voice."""
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')

# Load the pre-trained model for face recognition
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Initialize KNN classifier and fit the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Background image
imgBackground = cv2.imread("bg.png")

# Column names for the attendance CSV
COL_NAMES = ['NAME', 'DATE', 'TIME']

# Ensure that the Attendance directory exists
attendance_folder = "Attendance"
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

while True:
    # Get current timestamp and date before processing the frame
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")  # Get the current date
    attendance_file = os.path.join(attendance_folder, f"Attendance_{date}.csv")  # Define the attendance file path

    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]

        # Convert to grayscale for better matching (if necessary)
        gray_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_crop_img, (75, 50))  # Resize to 75x50 (grayscale)
        flattened_img = resized_img.flatten().reshape(1, -1)  # Flatten the image to 3750 features

        # Debugging: Check the shapes
        print(f"Shape of resized grayscale image: {resized_img.shape}")  # This should be (75, 50)
        print(f"Shape of flattened image: {flattened_img.shape}")  # This should be (1, 3750)

        # Predict using KNN
        output = knn.predict(flattened_img)

        # Get current timestamp
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Draw rectangles and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Store attendance information
        attendance = [str(output[0]), str(date), str(timestamp)]

    # Display the frame with background image
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)

    # Wait for user input
    k = cv2.waitKey(1)
    if k == ord('o'):  # When 'o' key is pressed, record attendance
        speak("Attendance Taken..")
        time.sleep(5)

        # Check if the CSV file already exists for the current date
        exist = os.path.isfile(attendance_file)

        # Write the attendance data to the CSV file
        with open(attendance_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  # Write column names if the file is new
            writer.writerow(attendance)  # Append the attendance

    if k == ord('q'):  # When 'q' key is pressed, exit the loop
        break

# Release resources and close windows
video.release()
cv2.destroyAllWindows()
