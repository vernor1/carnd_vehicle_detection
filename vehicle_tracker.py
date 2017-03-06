import cv2
import numpy as np

from feature_extraction import GetHogFeatures, GetSpatialFeatures, GetColorHistFeatures
from scipy.ndimage.measurements import label


# Helper Functions ------------------------------------------------------------
def GetHeatMap(img, boundingBoxes, threshold=0):
    """ Generates a thresholded heat map for given bounding boxes.

    param: img: Source image, only used to determine the heat map size
    param: boundingBoxes: A sequence of bounding boxes
    param: threshold: Heat threshold
    returns: The heat map
    """
    heat = np.zeros_like(img[:,:,0]).astype(np.uint32)
    for box in boundingBoxes:
        # Add 1 for all pixels inside each box, assuming each box takes the form ((x1, y1), (x2, y2))
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Zero out pixels below the threshold
    heat[heat <= threshold] = 0
    return heat

def GetBoxCenter(box):
    """ Computes coordinates of the center of a box.

    param: box: The box coordinates in form of ((x1,y1),(x2,y2)), where the first tuple is the coordinates of the left top corner of the box,
                the second tuple is the coordinates of the right bottom corner.
    returns: Center coordinates (x,y)
    """
    return box[0][0]+(box[1][0]-box[0][0])//2, box[0][1]+(box[1][1]-box[0][1])//2

def GetDistance(p1, p2):
    """ Computes the distance between two points.

    param: p1: The first point coordinates (x,y)
    param: p2: The second point coordinates (x,y)
    returns: Distance between the two points
    """
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class TVehicleTracker():
    """ Vehicle tracker class.
    """
    class TVehicle():
        # Constants -----------------------------------------------------------
        # Max bounding box history length
        MAX_HISTORY_LENGTH = 12
        # Max distance between the average bounding box center and any matching bounding box
        MAX_BOX_CENTER_DISTANCE = 128

        # Public Members ------------------------------------------------------
        def __init__(self, box):
            """ TVehicle ctor.

            param: box: Initial bounding box
            """
            # Vehicle Id
            self.Id = 0
            # History
            self.BoundingBoxHistory = [box]
            # Indication if the vehicle is locked
            self.IsLocked = False

        def SetId(self, vehicleId):
            """ Sets the vehicle Id.

            param: vehicleId: Vehicle Id
            """
            self.Id = vehicleId

        def GetId(self):
            """ Gets the vehicle Id.

            returns: Vehicle Id
            """
            return self.Id

        def GetLockState(self):
            """ Gets the current lock state.

            returns: "LOCKED": The object is in locked state, the history length is >0
                     "LOCKING": Indicating an unlocked state, but the history is collecting
                     "UNLOCKED": The history is empty, the object can be released
            """
            if self.IsLocked:
                return "LOCKED"
            elif len(self.BoundingBoxHistory) > 0:
                return "LOCKING"
            return "UNLOCKED"

        def GetBoundingBox(self):
            """ Gets the bounding box of the vehicle.

            returns: Coordinates of left top and right bottom corners of the bounding box ((x1,y1),(x2,y2))
            """
            if len(self.BoundingBoxHistory) == 0:
                return None
            return np.average(self.BoundingBoxHistory, axis=0).astype(np.uint16)

        def Update(self, boundingBoxes):
            """ Updates the vehicle with new bounding boxes.

            param: boundingBoxes: New bounding boxes to search for matching ones
            returns: Index of the best matching bounding box, otherwise None
            """
            if len(self.BoundingBoxHistory) == 0:
                return None
            averageBoxCenter = GetBoxCenter(self.GetBoundingBox())
            matchingBoxes = {}
            for idx in range(len(boundingBoxes)):
                # Compute the distance between the center of new box and the center of own average box
                distance = GetDistance(GetBoxCenter(boundingBoxes[idx]), averageBoxCenter)
                # The new box is matching if the distance is below the threshold
                if distance <= self.MAX_BOX_CENTER_DISTANCE:
                    matchingBoxes[distance] = idx
            processedIdx = None
            if len(matchingBoxes) == 0:
                # If no boxes detected for the vehicle, reduce the history
                if len(self.BoundingBoxHistory) > 0:
                    self.BoundingBoxHistory.pop(0)
                print("Updated vehicle Id %d, history length %d" % (self.Id, len(self.BoundingBoxHistory)))
                if len(self.BoundingBoxHistory) == 0:
                    # If the history is empty, consider the vehicle no longer locked
                    self.IsLocked = False
            else:
                # Find the best match
                processedIdx = matchingBoxes[sorted(matchingBoxes)[0]]
                self.BoundingBoxHistory.append(boundingBoxes[processedIdx])
                # Truncate the history if necessary
                if len(self.BoundingBoxHistory) > self.MAX_HISTORY_LENGTH:
                    self.BoundingBoxHistory.pop(0)
                print("Updated vehicle Id %d, history length %d, best box %s" % (self.Id,
                                                                                 len(self.BoundingBoxHistory),
                                                                                 boundingBoxes[processedIdx],))
                if len(self.BoundingBoxHistory) == self.MAX_HISTORY_LENGTH:
                    # Once the history is full, the vehicle is locked
                    self.IsLocked = True
            return processedIdx

    # Constants ---------------------------------------------------------------
    PX_PER_CELL = 8
    CELLS_PER_STEP = 2
    WINDOW_SIZE_PX = 64
    SCALES = [1.0, 1.5, 2.0]
    # Max heat map history length
    MAX_HISTORY_LENGTH = 12
    MAX_VEHICLE_ID = 20
    # Min dimensions of a valid bounding box
    MIN_BOX_WIDTH = 48
    MIN_BOX_HEIGHT = 48

    # Public Members ----------------------------------------------------------
    def __init__(self, classifier, rangeY):
        """ TVehicleTracker ctor.

        param: classifier: TClassifier instance
        param: rangeY: Tuple of min and max vertical coordinates for sliding windows
        """
        self.Classifier = classifier
        self.RangeY = rangeY
        self.HeatMapHistory = []
        # Set of used vehicle Ids
        self.VehicleIds = set()
        # List of TVehicle instances
        self.Vehicles = []

    def ProcessImage(self, img):
        """ Processes an image of the road.

        param: img: Road image
        returns: Bounding boxes where the classifier detected a vehicle (for debugging purposes)
        """
        boundingBoxes = self.GetBoundingBoxes(img)
        heatMap = GetHeatMap(img, boundingBoxes)
        self.HeatMapHistory.append(heatMap)
        # Truncate the history if the max length reached
        if len(self.HeatMapHistory) > self.MAX_HISTORY_LENGTH:
            self.HeatMapHistory.pop(0)
        self.UpdateVehicles()
        return boundingBoxes

    def GetVehicles(self):
        """ Gets detected vehicles.

        returns: Tuple of vehicle Ids and bounding boxes ((x1,y1),(x2,y2))
        """
        vehicleIds = []
        boundingBoxes = []
        for vehicle in self.Vehicles:
            if vehicle.GetLockState() == "LOCKED":
                vehicleIds.append(vehicle.GetId())
                boundingBoxes.append(vehicle.GetBoundingBox())
        return vehicleIds, boundingBoxes

    # Private Members ---------------------------------------------------------
    def GetBoundingBoxes(self, img):
        """ Gets all detected bounding boxes for an image.

        param: img: Road image
        returns: All bounding boxes where the classifier detected a vehicle
        """
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
            ycrcbImg = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2YCrCb)
            nrOfBlocksX = scaledImg.shape[1] // self.PX_PER_CELL - 1
            nrOfBlocksY = scaledImg.shape[0] // self.PX_PER_CELL - 1
            windowSizeBlocks = self.WINDOW_SIZE_PX // self.PX_PER_CELL - 1
            nrOfStepsX = (nrOfBlocksX - windowSizeBlocks) // self.CELLS_PER_STEP
            nrOfStepsY = (nrOfBlocksY - windowSizeBlocks) // self.CELLS_PER_STEP
            hogCh1 = GetHogFeatures(ycrcbImg[:,:,0], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
            hogCh2 = GetHogFeatures(ycrcbImg[:,:,1], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
            hogCh3 = GetHogFeatures(ycrcbImg[:,:,2], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
            for blockNrY in range(nrOfStepsY):
                for blockNrX in range(nrOfStepsX):
                    posX = blockNrX * self.CELLS_PER_STEP
                    posY = blockNrY * self.CELLS_PER_STEP
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
        """ Gets labels for the heat map history.

        returns: Labeled array
        """
        # Make an integral heat map out from the historical heat maps
        heatMap = np.sum(self.HeatMapHistory, axis=0)
        # Zero out pixels below the threshold of history length
        heatMap[heatMap <= len(self.HeatMapHistory)] = 0
        labeledArray, numFeatures = label(heatMap)
        return labeledArray, numFeatures

    def GetHistoricalBoundingBoxes(self):
        """ Gets historical bounding boxes.

        returns: boundingBoxes: A sequence of bounding boxes
        """
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

    def RemoveInvalidBoxes(self, boundingBoxes):
        """ Removes invalid bounding boxes.

        param: boundingBoxes: A sequence of bounding boxes
        returns: Only valid bounding boxes
        """
        validBoxes = []
        for box in boundingBoxes:
            width = box[1][0] - box[0][0]
            height = box[1][1] - box[0][1]
            if width >= self.MIN_BOX_WIDTH and height >= self.MIN_BOX_HEIGHT:
                validBoxes.append(box)
            else:
                print("Discard invalid box %s" % (box,))
        return validBoxes

    def CreateVehicleId(self):
        """ Creates a unique vehicle Id and updates the list of them.

        returns: New vehicle Id
        """
        for vehicleId in range(1, self.MAX_VEHICLE_ID):
            if vehicleId not in self.VehicleIds:
                print("Creating vehicle Id %d" % (vehicleId))
                self.VehicleIds.add(vehicleId)
                return vehicleId

    def CreateVehicle(self, box):
        """ Creates an instance of TVehicle.

        param: box: Initial bounding box
        """
        print("Creating nameless vehicle object")
        self.Vehicles.append(self.TVehicle(box))

    def UpdateVehicles(self):
        """ Updates, creates or deletes vehicle objects.
        """
        # Get all historical bounding boxes including possible false positives
        boundingBoxes = self.GetHistoricalBoundingBoxes()
        # Remove invalid boxes
        boundingBoxes = self.RemoveInvalidBoxes(boundingBoxes)
        processedBoxIndices = set()
        for vehicle in self.Vehicles:
            # Ask every vehicle object to process bounding boxes
            processedIdx = vehicle.Update(boundingBoxes)
            if processedIdx != None:
                processedBoxIndices.add(processedIdx)
        # Iterate through historical bounding boxes
        for idx in range(len(boundingBoxes)):
            if idx not in processedBoxIndices:
                # The valid box is not processed by any vehicle, create a new vehicle
                self.CreateVehicle(boundingBoxes[idx])
        unlockedVehicleIndices = []
        for idx in range(len(self.Vehicles)):
            # Notify vehicles of completing the update
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
                unlockedVehicleIndices.append(idx)
        for idx in unlockedVehicleIndices:
            # Remove unlocked vehicles
            del self.Vehicles[idx]
            print("Vehicle list length %d" % (len(self.Vehicles)))
