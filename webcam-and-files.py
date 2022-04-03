import cv2
from model import FacialExpressionModel
import numpy as np
import logging
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def render(image):
    gray_fr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    logging.info("Faces detected")
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(image, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('Image', image)
    
# For static images:
IMAGE_FILES = []
for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    render(image)
    
#for video files
video_files = []
for idx, file in enumerate(video_files):
    video = cv2.VideoCapture(file)
    while video.isOpened():
        success, image = cap.read()
        render(image)
        
# For webcam input:
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    render(image)    
    if cv2.waitKey(5) & 0xFF == 27:
      break
