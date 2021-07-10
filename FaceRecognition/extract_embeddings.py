# Here we have used OpenCV packages for Facial Recognition using Deep Learning
# This file contains the program to make embeddings of face , first the face is detected using face_detector_model then its is used to make embeddings which are saved and used for comparing.
# Use the following command in the command line argument by inputing location of the image where your file exists. 
# python extract_embeddings.py --dataset C:/Users/devam/Desktop/FaceRecognition/dataset --embeddings C:/Users/devam/Desktop/FaceRecognition/output/embeddings.pickle --detector C:/Users/devam/Desktop/FaceRecognition/face_detector_model --embedding-model C:/Users/devam/Desktop/FaceRecognition/openface.nn4.small2.v1.t7
# We hae used caffemodel which is a 152 layer pre trained CNN module and more accurate than haar cascades.
# openface_nn4.small2.v1.t7 : A Torch deep learning model which produces the 128-D facial embeddings.


        
# Import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


# Construct the argument parse and parse the arguments which can be used in command line interface to load the data
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# Get current directory
BASE_DIR = os.path.dirname(__file__)
print("[INFO] BASE DIR: ", BASE_DIR)


# load our face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)                  # Read the network model from .caffemodel format

# load our face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])               # Reads a network model stored in Torch7 framework's format


# Grabbing the paths of dataset input images
print("[INFO] Load image dataset..")
imagePaths = list(paths.list_images(args["dataset"]))
print("[DEBUG] Image Paths: ", imagePaths)

# Initialize list of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []

# Initialize the total number of faces processed
total = 0

# Loop over the image paths
# enumerate() is used to keep the count of iterations
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	# load the image, resize it to have a width of 600 pixels and then grab the image dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
    #.blobFromImage take the input as an image in our dnn model and used for preprocessing images
    # swapRB - flag which indicates that swap first and last channels in 3-channel image is necessary.
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)    
	# apply deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]                                     # extract the confidence (i.e., probability) associated with the prediction
		print("Confidence: ", confidence)

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])          # Multiplyijg with width and height to get real image as it was normalised earlier
			(startX, startY, endX, endY) = box.astype("int")
			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			# add the name of the person + corresponding face embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()




#Reference : https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
