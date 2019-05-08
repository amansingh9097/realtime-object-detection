#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:38:02 2018

@author: Aman Singh

USAGE: python build_face_dataset_live.py -c /home/aman/Documents/cv2/data/haarcascade_frontalface_default.xml -o image_dataset/aman
"""
# loading libraries
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# argparser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help = "path where the Haar cascade frontalface default xml file resides")
ap.add_argument("-o", "--output", required=True,
                help = "path where the images dataset will be stored")
args = vars(ap.parse_args())

# loading OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(args["cascade"])

# initializing video stream
# also initialize total no of example faces written to disk
print("[INFO] starting video stream....")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()

time.sleep(2.0)
total = 0

# looping over frames from the video stream
while True:
    """
    task:
        grab the frames from the threaded video stream, clone it
        (just in case we want to write it to disk), and then
        resize the frames so we can apply face detection faster
    """
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)
    
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
            )
    
    # loop over the face detections and draw them on the frame
    for (x,y,w,h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if key 'k' pressed, write original frame to disk
    if key == ord("k"):
        p = os.path.sep.join(
                [args["output"], "{}.png".format(
                        str(total).zfill(5))
                ]
        )
        cv2.imwrite(p, orig)
        total += 1
        
    # if 'q' pressed, exit out
    elif key == ord("q"):
        break
