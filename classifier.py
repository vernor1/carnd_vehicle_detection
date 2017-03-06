import cv2
import glob
import numpy as np
import os.path
import pickle

from feature_extraction import ExtractFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class TClassifier():
    """ Vehicle/non-vehicle classifier class.
    """
    # Constants ---------------------------------------------------------------
    TRAINING_SET_FILE = "training_set.p"
    CLASSIFIER_DATA_FILE = "classifier_data.p"

    # Public Members ----------------------------------------------------------
    def __init__(self, vehiclesDirectory, nonVehiclesDirectory):
        """ TClassifier ctor.

        param: vehiclesDirectory: Directory containing vehicle images
        param: vehiclesDirectory: Directory containing non-vehicle images
        """
        self.VehiclesDirectory = vehiclesDirectory
        self.NonVehiclesDirectory = nonVehiclesDirectory
        classifierData = {}
        if os.path.isfile(self.CLASSIFIER_DATA_FILE):
            print("Loading classifier data")
            classifierData = pickle.load(open(self.CLASSIFIER_DATA_FILE, "rb"))
            self.Scaler = classifierData["Scaler"]
            self.Classifier = classifierData["Classifier"]
        else:
            trainingSet = {}
            featureList = []
            labelList = []
            if os.path.isfile(self.TRAINING_SET_FILE):
                print("Loading training set")
                trainingSet = pickle.load(open(self.TRAINING_SET_FILE, "rb"))
                featureList = trainingSet["featureList"]
                labelList = trainingSet["labelList"]
                print("Loaded %d training samples" % (len(labelList)))
            else:
                print("Searching and extracting features of vehicle samples")
                for fileName in glob.iglob("%s/**/*.png" % (self.VehiclesDirectory), recursive=True):
                    featureList.append(ExtractFeatures(cv2.imread(fileName)))
                    labelList.append(1)
                print("Number of training samples of vehicles %d" % (len(labelList)))
                print("Searching and extracting features of non-vehicle samples")
                for fileName in glob.iglob("%s/**/*.png" % (self.NonVehiclesDirectory), recursive=True):
                    featureList.append(ExtractFeatures(cv2.imread(fileName)))
                    labelList.append(0)
                print("Total number of training samples %d" % (len(labelList)))
                print("Saving samples for fast access")
                trainingSet["featureList"] = featureList
                trainingSet["labelList"] = labelList
                pickle.dump(trainingSet, open(self.TRAINING_SET_FILE, "wb"))
            print("Creating feature scaler")
            # Fit a per-column scaler
            self.Scaler = StandardScaler().fit(featureList)
            # Apply the scaler
            print("Normalizing feature list")
            X_total = self.Scaler.transform(featureList)
            y_total = np.asarray(labelList, dtype=np.uint8)
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, random_state=rand_state)
            print("Training classifier")
            # Use a linear SVC (support vector classifier)
            self.Classifier = LinearSVC()
            # Train the SVC
            self.Classifier.fit(X_train, y_train)
            print("Test accuracy %.4f" % (self.Classifier.score(X_test, y_test)))
            print("Saving classifier data for fast access")
            classifierData["Scaler"] = self.Scaler
            classifierData["Classifier"] = self.Classifier
            pickle.dump(classifierData, open(self.CLASSIFIER_DATA_FILE, "wb"))

    def Normalize(self, featureList):
        """ Normalizes a list of feature vectors by-column. The method must be applied to test features before predicting labels.

        param: featureList: List of feature vectors
        returns: Normalized feature vectors
        """
        return self.Scaler.transform(featureList)

    def Predict(self, featureList):
        """ Predicts labels of the given feature vectors.

        param: featureList: List of feature vectors
        returns: List of labels
        """
        return self.Classifier.predict(featureList)
