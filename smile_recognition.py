# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 01:34:57 2019

@author: Ravi
"""

#Face Recognition using OpenCV

# importing the libraries
import cv2

#Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining function that will do detections
def detect(gray, frame):
     #returns x,y,w,h where x,y is top-left coordinate w is width and h is height
     #1.3 is howmuch we need compress image and 5 represent how many neighbour image we consider
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for x, y, w , h in faces:
         #drawingrectangle for face
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,165,0), 2)
         #region or zone of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for sx,sy, sw, sh in smiles:
            #drawing rectangle gor smile
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,255,0), 2)
    return frame
# Doing some face recognition with webcam
    #0 to use internal web cam and 1 for use external
video_capture = cv2.VideoCapture(0)
while True:
    #read method return 2 objects we are interested in only one that is last frame so we get only that
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#release web cam
video_capture.release()
#destroy all capture frames
cv2.destroyAllWindows()