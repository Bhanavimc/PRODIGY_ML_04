import cv2  # Import OpenCV
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np  # Make sure to import numpy

model_path = 'gesture_recognizer.task'  # Ensure this path is correct
BaseOptions = python.BaseOptions

# Define the gesture recognizer and its options
options = vision.GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE)

# Create a gesture recognizer instance
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    # Load the input image from an image file
    mp_image = mp.Image.create_from_file('WIN_20241104_21_38_01_Pro.jpg')

    # Perform gesture recognition on the provided image
    gesture_recognition_result = recognizer.recognize(mp_image)
    print('Gesture recognition result:', gesture_recognition_result)
