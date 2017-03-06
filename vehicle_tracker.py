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

def GetBoxCenter(box):
    return box[0][0]+(box[1][0]-box[0][0])//2, box[0][1]+(box[1][1]-box[0][1])//2

def GetDistance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class TVehicleTracker():
    """ Vehicle tracker class.
    """
    class TVehicle():
        # Constants -----------------------------------------------------------
        MAX_HISTORY_LENGTH = 12
        MAX_BOX_CENTER_DISTANCE = 128

        # Public Members ------------------------------------------------------
        def __init__(self, box):
            self.Id = 0
            self.BoundingBoxHistory = [box]
            self.IsLocked = False
            self.IsUpdated = True

        def SetId(self, vehicleId):
            self.Id = vehicleId

        def GetId(self):
            return self.Id

        def GetLockState(self):
            if self.IsLocked:
                # In locked state the history length is >0
                return "LOCKED"
            elif len(self.BoundingBoxHistory) > 0:
                # The locking state is indicating an unlocked state, but the history is collecting
                return "LOCKING"
            # Otherwise the vehicle is unlocked and the history is empty, the object can be released
            return "UNLOCKED"

        def GetBoundingBox(self):
            if len(self.BoundingBoxHistory) == 0:
                return None
            return np.average(self.BoundingBoxHistory, axis=0).astype(np.uint16)

        def Update(self, box):
            if len(self.BoundingBoxHistory) == 0:
                return False
            averageBox = self.GetBoundingBox()
            # Compute the distance between the center of new box and the center of own average box
            distance = GetDistance(GetBoxCenter(box), GetBoxCenter(averageBox))
            # The new box is accepted if the distance is below the threshold
            if distance <= self.MAX_BOX_CENTER_DISTANCE:
                self.BoundingBoxHistory.append(box)
                self.IsUpdated = True
                # Truncate the history if necessary
                if len(self.BoundingBoxHistory) > self.MAX_HISTORY_LENGTH:
                    self.BoundingBoxHistory.pop(0)
                print("Vehicle updated, history length %d" % (len(self.BoundingBoxHistory)))
                return True
            return False

        def CompleteUpdate(self):
            if not self.IsUpdated:
                # If no boxes detected for the vehicle, reduce the history
                if len(self.BoundingBoxHistory) > 0:
                    self.BoundingBoxHistory.pop(0)
                print("Reduced history length %d" % (len(self.BoundingBoxHistory)))
            if len(self.BoundingBoxHistory) == 0:
                # If the history is empty, consider the vehicle no longer locked
                self.IsLocked = False
            elif len(self.BoundingBoxHistory) == self.MAX_HISTORY_LENGTH:
                # Once the history is full, the vehicle is locked
                self.IsLocked = True
            # Reset the flag in the end of the update transaction
            self.IsUpdated = False

    # Constants ---------------------------------------------------------------
    PX_PER_CELL = 8
    WINDOW_SIZE_PX = 64
    SCALES = [1.0, 1.5, 2.0]
    MAX_HISTORY_LENGTH = 12
    MAX_VEHICLE_ID = 20
    MIN_BOX_WIDTH = 48
    MIN_BOX_HEIGHT = 48

    # Public Members ----------------------------------------------------------
    def __init__(self, classifier, rangeY):
        """ TVehicleTracker ctor.
        """
        self.Classifier = classifier
        self.RangeY = rangeY
        self.HeatMapHistory = []
        self.VehicleIds = set()
        self.Vehicles = []

    def ProcessImage(self, img):
        boundingBoxes = self.GetBoundingBoxes(img)
        heatMap = GetHeatMap(img, boundingBoxes)
        self.HeatMapHistory.append(heatMap)
        # Truncate the history if the max length reached
        if len(self.HeatMapHistory) > self.MAX_HISTORY_LENGTH:
            self.HeatMapHistory.pop(0)
#        print("Heat map history length %d" % (len(self.HeatMapHistory)))
        self.UpdateVehicles()
        return boundingBoxes

    def GetVehicles(self):
        vehicleIds = []
        boundingBoxes = []
        for vehicle in self.Vehicles:
            if vehicle.GetLockState() == "LOCKED":
                vehicleIds.append(vehicle.GetId())
                boundingBoxes.append(vehicle.GetBoundingBox())
        return vehicleIds, boundingBoxes

    # Private Members ---------------------------------------------------------
    def GetBoundingBoxes(self, img):
        boundingBoxes = []
        # Crop the image to the area of interest
        img = img[self.RangeY[0]:self.RangeY[1], :, :]
        normImg = np.empty_like(img)
        cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        for scale in self.SCALES:
            if scale == 1:
                scaledImg = np.copy(normImg)
            else:
                scaledImg = cv2.resize(normImg, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)))
            if scale > 1.5:
                # TODO: consider making it a constant
                cellsPerStep = 2
            else:
                cellsPerStep = 2
            ycrcbImg = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2YCrCb)
            nrOfBlocksX = scaledImg.shape[1] // self.PX_PER_CELL - 1
            nrOfBlocksY = scaledImg.shape[0] // self.PX_PER_CELL - 1
            windowSizeBlocks = self.WINDOW_SIZE_PX // self.PX_PER_CELL - 1
            nrOfStepsX = (nrOfBlocksX - windowSizeBlocks) // cellsPerStep
            nrOfStepsY = (nrOfBlocksY - windowSizeBlocks) // cellsPerStep
            hogCh1 = GetHogFeatures(ycrcbImg[:,:,0], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
            hogCh2 = GetHogFeatures(ycrcbImg[:,:,1], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
            hogCh3 = GetHogFeatures(ycrcbImg[:,:,2], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
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
                    spatialFeatures = GetSpatialFeatures(ycrcbImg[windowCoordY:windowCoordY + self.WINDOW_SIZE_PX,
                                                                  windowCoordX:windowCoordX + self.WINDOW_SIZE_PX],
                                                         size=(32, 32))
                    # Extract color histogram features of the patch
                    histFeatures = GetColorHistFeatures(ycrcbImg[windowCoordY:windowCoordY + self.WINDOW_SIZE_PX,
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

    def GetLabels(self):
        # Make an integral heat map out from the historical heat maps
        heatMap = np.sum(self.HeatMapHistory, axis=0)
        # Zero out pixels below the threshold of history length
        heatMap[heatMap <= len(self.HeatMapHistory)] = 0
        labeledArray, numFeatures = label(heatMap)
        return labeledArray, numFeatures

    def GetHistoricalBoundingBoxes(self):
        boundingBoxes = []
        labeledArray, numFeatures = self.GetLabels()
        # Iterate through all detected vehicles
        for idx in range(1, numFeatures + 1):
            # Find pixels with each vehicle Id value
            nonZero = (labeledArray == idx).nonzero()
            # Identify x and y values of those pixels
            nonZeroY = np.array(nonZero[0])
            nonZeroX = np.array(nonZero[1])
            # Define a bounding box based on min/max x and y
            box = (np.min(nonZeroX), np.min(nonZeroY)), (np.max(nonZeroX), np.max(nonZeroY))
            boundingBoxes.append(box)
        return boundingBoxes

    def IsBoxValid(self, box):
        width = box[1][0] - box[0][0]
        height = box[1][1] - box[0][1]
#        print("Box width %d, height %d" % (width, height))
        return width >= self.MIN_BOX_WIDTH and height >= self.MIN_BOX_HEIGHT

    def CreateVehicleId(self):
        for vehicleId in range(1, self.MAX_VEHICLE_ID):
            if vehicleId not in self.VehicleIds:
                print("Creating vehicle Id %d" % (vehicleId))
                self.VehicleIds.add(vehicleId)
                return vehicleId

    def CreateVehicle(self, box):
        print("Creating nameless vehicle object")
        self.Vehicles.append(self.TVehicle(box))

    def UpdateVehicles(self):
        boundingBoxes = self.GetHistoricalBoundingBoxes()
        # Iterate through historical bounding boxes
        for box in boundingBoxes:
            if not self.IsBoxValid(box):
                print("Discard invalid box %s" % (box,))
                continue
            isBoxAccepted = False
            for vehicle in self.Vehicles:
                if vehicle.Update(box):
                    isBoxAccepted = True
            # The valid box is not accepted by any vehicle, create a new vehicle
            if not isBoxAccepted:
                self.CreateVehicle(box)
        unlockedIndices = []
        for idx in range(len(self.Vehicles)):
            # Notify vehicles of completing the update
            self.Vehicles[idx].CompleteUpdate()
            vehicleId = self.Vehicles[idx].GetId()
            lockState = self.Vehicles[idx].GetLockState()
            print("Checking vehicle Id %d, state %s" % (vehicleId, lockState))
            if lockState == "LOCKED":
                if vehicleId == 0:
                    self.Vehicles[idx].SetId(self.CreateVehicleId())
                    print("Vehicle Ids %s" % (self.VehicleIds))
            elif lockState == "UNLOCKED":
                print("Vehicle Id %d is unlocked and to be removed" % (vehicleId))
                if vehicleId != 0:
                    self.VehicleIds.remove(vehicleId)
                    print("Vehicle Ids %s" % (self.VehicleIds))
                unlockedIndices.append(idx)
        for idx in unlockedIndices:
            # Remove unlocked vehicles
            del self.Vehicles[idx]
            print("Vehicle list length %d" % (len(self.Vehicles)))
