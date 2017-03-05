import cv2
import numpy as np

from feature_extraction import GetHogFeatures, GetSpatialFeatures, GetColorHistFeatures
from scipy.ndimage.measurements import label


# Helper Functions ------------------------------------------------------------
def GetHeatMap(img, boundingBoxes, threshold=0):
    heat = np.zeros_like(img[:,:,0]).astype(np.uint32)
    for box in boundingBoxes:
        # Add 1 for all pixels inside each box, assuming each box takes the form ((x1, y1), (x2, y2))
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Zero out pixels below the threshold
    heat[heat <= threshold] = 0
    return heat

class TVehicleTracker():
    """ Vehicle tracker class.
    """

    # Constants ---------------------------------------------------------------
    PX_PER_CELL = 8
    WINDOW_SIZE_PX = 64
    SCALES = [1.0, 1.5, 2.0]
    MAX_HISTORY_LENGTH = 7

    # Public Members ----------------------------------------------------------
    def __init__(self, classifier, rangeY):
        """ TVehicleTracker ctor.
        """
        self.Classifier = classifier
        self.RangeY = rangeY
        self.HeatMapHistory = []

    def ProcessImage(self, img):
        boundingBoxes = self.GetBoundingBoxes(img)
        heatMap = GetHeatMap(img, boundingBoxes, threshold=0)
        self.HeatMapHistory.append(heatMap)
        # Truncate the history if the max length reached
        if len(self.HeatMapHistory) > self.MAX_HISTORY_LENGTH:
            self.HeatMapHistory.pop(0)
        return boundingBoxes

    # Private Members ---------------------------------------------------------
    def GetBoundingBoxes(self, img):
        boundingBoxes = []
        # Crop the image to the area of interest
        img = img[self.RangeY[0]:self.RangeY[1], :, :]
        normImg = np.empty_like(img)
        cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        for scale in self.SCALES:
            print("Processing scale %.2f" % (scale))
            if scale == 1:
                scaledImg = np.copy(normImg)
            else:
                scaledImg = cv2.resize(normImg, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)))
            if scale > 1.5:
                cellsPerStep = 2
            else:
                cellsPerStep = 2
            hsvImg = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2HSV)
            hlsImg = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2HLS)
            nrOfBlocksX = scaledImg.shape[1] // self.PX_PER_CELL - 1
            nrOfBlocksY = scaledImg.shape[0] // self.PX_PER_CELL - 1
            windowSizeBlocks = self.WINDOW_SIZE_PX // self.PX_PER_CELL - 1
            nrOfStepsX = (nrOfBlocksX - windowSizeBlocks) // cellsPerStep
            nrOfStepsY = (nrOfBlocksY - windowSizeBlocks) // cellsPerStep
            hogCh1 = GetHogFeatures(hsvImg[:,:,0], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
            hogCh2 = GetHogFeatures(hsvImg[:,:,1], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
            hogCh3 = GetHogFeatures(hsvImg[:,:,2], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
            for blockNrY in range(nrOfStepsY):
                for blockNrX in range(nrOfStepsX):
                    posX = blockNrX * cellsPerStep
                    posY = blockNrY * cellsPerStep
                    # Extract HOG features of the patch
                    hogFeatures1 = hogCh1[posY:posY+windowSizeBlocks, posX:posX+windowSizeBlocks].ravel()
                    hogFeatures2 = hogCh2[posY:posY+windowSizeBlocks, posX:posX+windowSizeBlocks].ravel()
                    hogFeatures3 = hogCh3[posY:posY+windowSizeBlocks, posX:posX+windowSizeBlocks].ravel()
                    hogFeatures = np.concatenate((hogFeatures1, hogFeatures2, hogFeatures3))
                    windowCoordX = posX * self.PX_PER_CELL
                    windowCoordY = posY * self.PX_PER_CELL
                    # Extract spatial features of the patch
                    spatialFeatures = GetSpatialFeatures(hsvImg[windowCoordY:windowCoordY + self.WINDOW_SIZE_PX,
                                                                windowCoordX:windowCoordX + self.WINDOW_SIZE_PX],
                                                         size=(32, 32))
                    # Extract color histogram features of the patch
                    histFeatures = GetColorHistFeatures(hlsImg[windowCoordY:windowCoordY + self.WINDOW_SIZE_PX,
                                                               windowCoordX:windowCoordX + self.WINDOW_SIZE_PX],
                                                        nrOfBins=32)
                    featureList = np.concatenate((hogFeatures, spatialFeatures, histFeatures)).reshape(1, -1)
                    featureList = self.Classifier.Normalize(featureList)
                    trueWindowCoordX = np.int(windowCoordX * scale)
                    trueWindowCoordY = np.int(windowCoordY * scale)
                    trueWindowSize = np.int(self.WINDOW_SIZE_PX * scale)
                    if self.Classifier.Predict(featureList) == 1:
                        boundingBoxes.append(((trueWindowCoordX, trueWindowCoordY + self.RangeY[0]),
                                             (trueWindowCoordX + trueWindowSize, trueWindowCoordY + trueWindowSize + self.RangeY[0])))
        return boundingBoxes

    def GetHeatMap(self):
        heatMap = np.sum(self.HeatMapHistory, axis=0)
        heatMap[heatMap <= len(self.HeatMapHistory)] = 0
        return heatMap

    def GetLabels(self):
        labeledArray, numFeatures = label(self.GetHeatMap())
        return labeledArray, numFeatures

    def GetVehicles(self):
        vehicleIds = []
        boundingBoxes = []
        labeledArray, numFeatures = self.GetLabels()
        # Iterate through all detected vehicles
        for vehicleId in range(1, numFeatures + 1):
            # Find pixels with each vehicle Id value
            nonZero = (labeledArray == vehicleId).nonzero()
            # Identify x and y values of those pixels
            nonZeroY = np.array(nonZero[0])
            nonZeroX = np.array(nonZero[1])
            # Define a bounding box based on min/max x and y
            box = (np.min(nonZeroX), np.min(nonZeroY)), (np.max(nonZeroX), np.max(nonZeroY))
            vehicleIds.append(vehicleId)
            boundingBoxes.append(box)
        return vehicleIds, boundingBoxes
