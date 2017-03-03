import cv2
import glob
import numpy as np
import os.path
import pickle

from feature_extraction import ExtractFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


TRAINING_SET_FILE = "training_set.p"
TRAINING_SAMPLES_FILE = "training_samples.p"

def TrainClassifier(X_train, y_train):
    # Use a linear SVC (support vector classifier)
    classifier = LinearSVC()
    # Train the SVC
    classifier.fit(X_train, y_train)
    return classifier

def GetTrainingSet(vehiclesDirectory, nonVehiclesDirectory):
    trainingSet = {}
    if os.path.isfile(TRAINING_SET_FILE):
        print("Loading training set")
        trainingSet = pickle.load(open(TRAINING_SET_FILE, "rb"))
        X_train = trainingSet["X_train"]
        y_train = trainingSet["y_train"]
        print("Loaded %d training samples" % (len(y_train)))
    else:
        featureList = []
        labelList = []
        print("Searching and extracting features of vehicle samples")
        for fileName in glob.iglob("%s/**/*.png" % (vehiclesDirectory), recursive=True):
            featureList.append(ExtractFeatures(cv2.imread(fileName)))
            labelList.append(1)
        print("Number of training samples of vehicles %d" % (len(labelList)))
        print("Searching and extracting features of non-vehicle samples")
        for fileName in glob.iglob("%s/**/*.png" % (nonVehiclesDirectory), recursive=True):
            featureList.append(ExtractFeatures(cv2.imread(fileName)))
            labelList.append(0)
        print("Total number of training samples %d" % (len(labelList)))
        y_train = np.asarray(labelList, dtype=np.uint8)
        print("Normalizing training samples")
        # Fit a per-column scaler
        scaler = StandardScaler().fit(featureList)
        # Apply the scaler
        X_train = scaler.transform(featureList)
        print("Saving the samples for fast access")
        trainingSet["X_train"] = X_train
        trainingSet["y_train"] = y_train
        pickle.dump(trainingSet, open(TRAINING_SET_FILE, "wb"))
    return X_train, y_train

def GetTrainingSamples(vehiclesDirectory, nonVehiclesDirectory):
    vehicles = []
    nonVehicles = []
    trainingSamples = {}
    if os.path.isfile(TRAINING_SAMPLES_FILE):
        print("Loading training samples")
        trainingSamples = pickle.load(open(TRAINING_SAMPLES_FILE, "rb"))
        vehicles = trainingSamples["vehicles"]
        nonVehicles = trainingSamples["non-vehicles"]
        print("Loaded %d samples of vehicles" % (len(vehicles)))
        print("Loaded %d samples of non-vehicles" % (len(nonVehicles)))
    else:
        print("Searching for vehicle samples")
        for fileName in glob.iglob("%s/**/*.png" % (vehiclesDirectory), recursive=True):
            vehicles.append(cv2.imread(fileName))
        print("Loaded %d samples" % (len(vehicles)))
        print("Searching for non-vehicle samples")
        for fileName in glob.iglob("%s/**/*.png" % (nonVehiclesDirectory), recursive=True):
            nonVehicles.append(cv2.imread(fileName))
        print("Loaded %d samples" % (len(nonVehicles)))
        # Save the samples for fast access
        trainingSamples["vehicles"] = vehicles
        trainingSamples["non-vehicles"] = nonVehicles
        pickle.dump(trainingSamples, open(TRAINING_SAMPLES_FILE, "wb"))
    return vehicles, nonVehicles
