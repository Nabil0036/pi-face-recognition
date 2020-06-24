from imutils import paths
import face_recognition
import argparse
import pickle 
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("-i","--dataset", required=True,help = "path to inputdirectory of faces")
ap.add_argument("-e","--encoding", required=True,help="path to serialized db of facial encodings")
ap.add_argument("-d","--detection-method",type=str,default="hog",help="face detection model to use : either 'hog' or 'cnn' ")

args = vars(ap.parse_args())

print(args)

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

knownEncodings = []
knownNames = []

for (i,imagepath) in enumerate(imagePaths):
    print("[INFO] processing image{}/{}".format(i+1,len(imagePaths)))

    name = imagepath.split("/")[1]

    image = cv2.imread(imagepath)
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model = args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)


print("[INFO] serializing encodings..")
data = {"encodings":knownEncodings,"names":knownNames}

f = open(args["encoding"],"wb")
f.write(pickle.dumps(data))
f.close()