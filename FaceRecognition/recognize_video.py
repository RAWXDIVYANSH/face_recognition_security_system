# Here we have used OpenCV packages for Facial Recognition using Deep Learning
# This file contains the program to recognize the face in real time vieo stream , first the face is detected using face_detector_model then its is used to make embeddings and finally it is used to recognnize by comparing with the test image embedding.
# Use the following command in the command line argument by inputing location of the image where your file exists. 
# python recognize.py --detector C:/Users/devam/Desktop/FaceRecognition/face_detector_model --embedding-model C:/Users/devam/Desktop/FaceRecognition/openface.nn4.small2.v1.t7 --recognizer C:/Users/devam/Desktop/FaceRecognition/output/recognizer.pickle --le C:/Users/devam/Desktop/FaceRecognition/output/le.pickle --confidence FLOAT_VALUE
# We hae used caffemodel which is a 152 layer pre trained CNN module and more accurate than haar cascades.


        
# Import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


# Construct the argument parse and parse the arguments which can be used in command line interface to load the data
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# load our serialized face detector from disk
print("Loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# load our serialized face embedding model from disk
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])                # Reads a network model stored in Torch7 framework's format

 
# load the actual face recognition model along with the label encoder
# rb : Opens the file as read-only in binary format and starts reading from the beginning of the file. While binary format can be used for different purposes, it is usually used when dealing with things like images, videos,
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read()) 


# Initialize the video stream, then allow the camera sensor to warm up
print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# FPS().start() will starts a timer that we can use to measure FPS, or more specifically, the throughput rate of our video processing pipeline.
fps = FPS().start()


# loop over frames from the video file stream
while True:
	# grab the frame fromm video stream
	frame = vs.read()
	# resize the frame to have a width of 600 pixels and then grab the image dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	# construct a blob from the image
    #.blobFromImage take the input as an image in our dnn model and used for preprocessing images
    # swapRB - flag which indicates that swap first and last channels in 3-channel image is necessary.
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# Apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()
    # loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence associated with the prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
            # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			if proba<0.5:
				continue
			name = le.classes_[j]
			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),(255, 255, 0), 1)
			cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
	# update the FPS counter
	fps.update()
    # show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# stop the timer and display FPS information
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()