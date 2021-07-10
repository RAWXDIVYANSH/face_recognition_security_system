# Here we are detecting faces with the help of opencv dnn package which contains a module face_detection.
# Use the following command in the command line argument by inputing location of the image where your file exists. 
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
# We hae used caffemodel which is a 152 layer pre trained CNN module and more accurate than haar cascades.
# Using argparse for command line input


# Importing packages
import os
import numpy as np
import argparse
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# Get current directory
dirname, filename = os.path.split(os.path.abspath(__file__))
prototxt = os.path.join(dirname, args["prototxt"])
model = os.path.join(dirname, args["model"])
image = os.path.join(dirname, args["image"])       


# Load model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)


# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(image)          # Taking image from the input argument of image
(h, w) = image.shape[:2]

#.blobFromImage take the input as an image in our dnn model
blob = cv2.dnn.blobFromImage(cv2.resize(image, (600, 500)), 1.0, (300,300), 
                             (104.0, 177.0, 123.0))              #(104.0, 177.0, 123.0) fixed and is used to normalise image     

# Here we pass the blob through the network and obtain predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward() 

# loop over the detections
for i in range(0, detections.shape[2]):                         #detections.shape[2] is used to get the no of detections
	# extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]            # Third item of the list which itself is the 4 item of detection list

	# Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
	if confidence >= args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the object
        # 3:7 is the [x,y,h,w]
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])     # Multiplyijg with width and height to get real image as it was normalised earlier
		(startX, startY, endX, endY) = box.astype("int")
 
		# Draw the bounding box of the face along with the associated probability
		text = f"{(confidence * 100):.2f}%"
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 255, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
