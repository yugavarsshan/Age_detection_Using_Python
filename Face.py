import cv2
import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a VideoCapture object to capture video from the webcam
video_capture = cv2.VideoCapture(0)

# Set the width and height of the video capture
video_capture.set(3, 640)
video_capture.set(4, 480)

# Loop over frames from the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = video_capture.read()

    # Convert the image from color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate the age of the person based on the face size
        face_size = max(w, h)
        age = int(face_size * 0.05)

        # Display the age on the frame
        cv2.putText(frame, "Age: " + str(age), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) == 27: # Escape key
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
