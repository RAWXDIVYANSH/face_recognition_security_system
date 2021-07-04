# Here we are detecting faces with the help of opencv dnn package which contains a module face_detection.
# Use the following command in the command line argument by inputing location of the image where your file exists. 
# python detect_faces_video.py  --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
# We hae used caffemodel which is a 152 layer pre trained CNN module and more accurate than haar cascades.
# Using argparse for command line input


# Importing packages
import os
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



# Get current directory
dirname, filename = os.path.split(os.path.abspath(__file__))
prototxt = os.path.join(dirname, args["prototxt"])
model = os.path.join(dirname, args["model"])

# Load model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Video stream frames are looped over
while True:
    # Video stream frame is resized it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
 
    # load the input video and construct an input blob for the image
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (500, 500)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
    #  Here we pass the blob through the network and obtain predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):                                    #detections.shape[2] is used to get the no of detections
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]                                   # Third item of the list which itself is the 4 item of detection list

        # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence < args["confidence"]:
                continue

        # compute the (x, y)-coordinates of the bounding box for the object
        # 3:7 is the [x,y,h,w]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])              # Multiplyijg with width and height to get real image as it was normalised earlier
        (startX, startY, endX, endY) = box.astype("int")
 
        # Draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()