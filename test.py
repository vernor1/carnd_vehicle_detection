import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from classifier import TClassifier
from feature_extraction import GetHogFeatures, GetSpatialFeatures, GetColorHistFeatures, ExtractFeatures
from vehicle_tracker import TVehicleTracker, GetHeatMap


# The following code is only used for debugging and generating test images
#------------------------------------------------------------------------------

TRAINING_SAMPLES_FILE = "training_samples.p"

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

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="Test Pipeline Components")
    argParser.add_argument("type",
                           choices=["class_examples", "feature_extraction", "vehicle_detection", "vehicle_tracking"])
    argParser.add_argument("--in_img",
                           type=str,
                           help="Path to the original image file")
    argParser.add_argument("--out_img",
                           type=str,
                           help="Path to the plot file of the applied transformation")
    args = argParser.parse_args()

    if args.type == "class_examples":
        vehicles, nonVehicles = GetTrainingSamples("vehicles", "non-vehicles")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
        fig.tight_layout()
        vehicleIdx = np.random.randint(len(vehicles))
        ax1.imshow(cv2.cvtColor(vehicles[vehicleIdx], cv2.COLOR_BGR2RGB))
        ax1.set_title("Vehicle", fontsize=20)
        nonVehicleIdx = np.random.randint(len(nonVehicles))
        ax2.imshow(cv2.cvtColor(nonVehicles[nonVehicleIdx], cv2.COLOR_BGR2RGB))
        ax2.set_title("Not Vehicle", fontsize=20)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        fig.savefig(args.out_img)

    elif args.type == "feature_extraction":
        vehicles, nonVehicles = GetTrainingSamples("vehicles", "non-vehicles")
        fig, ((ax1, ax2),
              (ax3, ax4),
              (ax5, ax6),
              (ax7, ax8),
              (ax9, ax10),
              (ax11, ax12),
              (ax13, ax14),
              (ax15, ax16),
              (ax17, ax18),
              (ax19, ax20)) = plt.subplots(10, 2, figsize=(10, 50))
        fig.tight_layout()
        imgIdx = np.random.randint(len(vehicles))
        img = vehicles[imgIdx]
        normImg = np.empty_like(img)
        cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        ax1.imshow(cv2.cvtColor(normImg, cv2.COLOR_BGR2RGB))
        ax1.set_title("Vehicle", fontsize=16)
        ycrcbImg = cv2.cvtColor(normImg, cv2.COLOR_BGR2YCrCb)
        hogFeatures, hogImgY = GetHogFeatures(ycrcbImg[:,:,0], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isVisualization=True)
        ax3.imshow(hogImgY, cmap="gray")
        ax3.set_title("Vehicle HOG Visualization (Y)", fontsize=16)
        hogFeatures, hogImgCr = GetHogFeatures(ycrcbImg[:,:,1], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isVisualization=True)
        ax5.imshow(hogImgCr, cmap="gray")
        ax5.set_title("Vehicle HOG Visualization (Cr)", fontsize=16)
        hogFeatures, hogImgCb = GetHogFeatures(ycrcbImg[:,:,2], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isVisualization=True)
        ax7.imshow(hogImgCb, cmap="gray")
        ax7.set_title("Vehicle HOG Visualization (Cb)", fontsize=16)
        spatialImg = GetSpatialFeatures(ycrcbImg, isFeatureVector=False)
        ax9.imshow(spatialImg[:,:,0], cmap="gray")
        ax9.set_title("Vehicle Spatial Features (Y)", fontsize=16)
        ax11.imshow(spatialImg[:,:,1], cmap="gray")
        ax11.set_title("Vehicle Spatial Features (Cr)", fontsize=16)
        ax13.imshow(spatialImg[:,:,2], cmap="gray")
        ax13.set_title("Vehicle Spatial Features (Cb)", fontsize=16)
        histFeatures, histY, histCr, histCb, binCenters = GetColorHistFeatures(ycrcbImg, nrOfBins=32, isVisualization=True)
        ax15.bar(binCenters, histY[0])
        ax15.set_title("Vehicle Color Hist. Features (Y)", fontsize=16)
        ax17.bar(binCenters, histCr[0])
        ax17.set_title("Vehicle Color Hist. Features (Cr)", fontsize=16)
        ax19.bar(binCenters, histCb[0])
        ax19.set_title("Vehicle Color Hist. Features (Cb)", fontsize=16)
        imgIdx = np.random.randint(len(nonVehicles))
        img = nonVehicles[imgIdx]
        normImg = np.empty_like(img)
        cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        ax2.imshow(cv2.cvtColor(normImg, cv2.COLOR_BGR2RGB))
        ax2.set_title("Not Vehicle", fontsize=16)
        ycrcbImg = cv2.cvtColor(normImg, cv2.COLOR_BGR2YCrCb)
        hogFeatures, hogImgY = GetHogFeatures(ycrcbImg[:,:,0], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isVisualization=True)
        ax4.imshow(hogImgY, cmap="gray")
        ax4.set_title("Non-vehicle HOG Visualization (Y)", fontsize=16)
        hogFeatures, hogImgCr = GetHogFeatures(ycrcbImg[:,:,1], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isVisualization=True)
        ax6.imshow(hogImgCr, cmap="gray")
        ax6.set_title("Non-vehicle HOG Visualization (Cr)", fontsize=16)
        hogFeatures, hogImgCb = GetHogFeatures(ycrcbImg[:,:,2], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isVisualization=True)
        ax8.imshow(hogImgCb, cmap="gray")
        ax8.set_title("Non-vehicle HOG Visualization (Cb)", fontsize=16)
        spatialImg = GetSpatialFeatures(ycrcbImg, isFeatureVector=False)
        ax10.imshow(spatialImg[:,:,0], cmap="gray")
        ax10.set_title("Non-vehicle Spatial Features (Y)", fontsize=16)
        ax12.imshow(spatialImg[:,:,1], cmap="gray")
        ax12.set_title("Non-vehicle Spatial Features (Cr)", fontsize=16)
        ax14.imshow(spatialImg[:,:,2], cmap="gray")
        ax14.set_title("Non-vehicle Spatial Features (Cb)", fontsize=16)
        histFeatures, histY, histCr, histCb, binCenters = GetColorHistFeatures(ycrcbImg, nrOfBins=32, isVisualization=True)
        ax16.bar(binCenters, histY[0])
        ax16.set_title("Non-vehicle Color Hist. Features (Y)", fontsize=16)
        ax18.bar(binCenters, histCr[0])
        ax18.set_title("Non-vehicle Color Hist. Features (Cr)", fontsize=16)
        ax20.bar(binCenters, histCb[0])
        ax20.set_title("Non-vehicle Color Hist. Features (Cb)", fontsize=16)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.01)
        fig.savefig(args.out_img)

    elif args.type == "vehicle_detection":
        testFiles = []
        for fileName in glob.glob(args.in_img):
            testFiles.append(fileName)
        classifier = TClassifier("vehicles", "non-vehicles")
        tracker = TVehicleTracker(classifier, (380, 660))
        fig, subplots = plt.subplots(len(testFiles), 2, figsize=(8, len(testFiles) * 2.5))
        fig.tight_layout()
        row = 0
        for filePath in testFiles:
            fileName = os.path.basename(filePath)
            print("Processing %s" % (fileName))
            img = cv2.imread(filePath)
            boundingBoxes = tracker.GetBoundingBoxes(img)
            for box in boundingBoxes:
                cv2.rectangle(img, box[0], box[1], (255, 0, 0), 3)
            subplots[row][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            subplots[row][0].set_title("Detections of %s" % (fileName), fontsize=12)
            subplots[row][1].imshow(np.clip(GetHeatMap(img, boundingBoxes), 0, 255), cmap="hot")
            subplots[row][1].set_title("Heat Map of %s" % (fileName), fontsize=12)
            row += 1
        plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02)
        fig.savefig(args.out_img)

    elif args.type == "vehicle_tracking":
        testFiles = []
        for fileName in glob.glob(args.in_img):
            testFiles.append(fileName)
        classifier = TClassifier("vehicles", "non-vehicles")
        tracker = TVehicleTracker(classifier, (380, 660))
        fig, subplots = plt.subplots(len(testFiles)+1, 2, figsize=(8, (len(testFiles)+1) * 2.5))
        fig.tight_layout()
        lastImg = None
        row = 0
        for filePath in testFiles:
            fileName = os.path.basename(filePath)
            print("Processing %s" % (fileName))
            img = cv2.imread(filePath)
            lastImg = np.copy(img)
            boundingBoxes = tracker.ProcessImage(img)
            for box in boundingBoxes:
                cv2.rectangle(img, box[0], box[1], (255, 0, 0), 3)
            subplots[row][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            subplots[row][0].set_title("Detections of %s" % (fileName), fontsize=12)
            subplots[row][1].imshow(np.clip(GetHeatMap(img, boundingBoxes), 0, 255), cmap="hot")
            subplots[row][1].set_title("Heat Map of %s" % (fileName), fontsize=12)
            row += 1
        vehicleIds, boundingBoxes = tracker.GetVehicles()
        print("%d vehicles found" % (len(vehicleIds)))
        for idx in range(len(vehicleIds)):
            boxLeftTop = boundingBoxes[idx][0][0], boundingBoxes[idx][0][1]
            boxRightBottom = boundingBoxes[idx][1][0], boundingBoxes[idx][1][1]
            cv2.rectangle(lastImg, boxLeftTop, boxRightBottom, (0, 255, 255), 3)
            cv2.putText(lastImg, "Vehicle %d" % (vehicleIds[idx]),
                        (boxLeftTop[0]+5, boxLeftTop[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        subplots[row][0].imshow(cv2.cvtColor(lastImg, cv2.COLOR_BGR2RGB))
        subplots[row][0].set_title("Bounding Boxes of Last Image", fontsize=12)
        labeledArray, numFeatures = tracker.GetLabels()
        subplots[row][1].imshow(labeledArray, cmap="gray")
        subplots[row][1].set_title("Resulting Labels", fontsize=12)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02)
        fig.savefig(args.out_img)
