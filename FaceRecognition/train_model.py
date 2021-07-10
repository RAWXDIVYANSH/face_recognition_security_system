# This is a training model file of our face recognition model.
# Here we have used OpenCV packages for Facial Recognition using Deep Learning
# This file contains the program to train the embeddings of face , first the face is detected using face_detector_model then its is used to make embeddings which are saved and used for training on our dataset.
# Use the following command in the command line argument by inputing location of the image where your file exists. 
# python train_model.py --embeddings C:/Users/devam/Desktop/FaceRecognition/output/embeddings.pickle --recognizer C:/Users/devam/Desktop/FaceRecognition/output/recognizer.pickle --le C:/Users/devam/Desktop/FaceRecognition/output/le.pickle
# We hae used caffemodel which is a 152 layer pre trained CNN module and more accurate than haar cascades.



# Import the necessary packages
# scikit is by far most used library for preprocessing different types of datasets.
# I have used Support vector classifier here
# pickel is used to convert a Python object into a byte stream to store it in a file/database
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())


# encode the labels
# converting the labels into numeric form so as to convert it into readable form.
print("[INFO] encoding labels...")
le = LabelEncoder()                               # LabelEncoder encode labels with a value between 0 and n_classes-1 where n is the number of distinct labels.
labels = le.fit_transform(data["names"])          # this method performs fit and transform on the input data at a single time and converts the data points.


# train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=7.0, kernel="linear", probability=True)       # use linear classifier to detect to which class our data belongs to
recognizer.fit(data["embeddings"], labels)


# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()


# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()