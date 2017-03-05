import cv2
import numpy as np

from feature_extraction import GetHogFeatures, GetSpatialFeatures, GetColorHistFeatures

PX_PER_CELL = 8
WINDOW_SIZE_PX = 64
CELLS_PER_STEP = 2
SCALES = [1.0, 1.5, 2.0]

def GetBoundingBoxes(classifier, img, rangeY):
    boundingBoxes = []
    # Crop the image to the area of interest
    img = img[rangeY[0]:rangeY[1], :, :]
    cv2.imwrite("tmp/cropped.png", img)
    normImg = np.empty_like(img)
    cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    for scale in SCALES:
        print("Processing scale %.2f" % (scale))
        if scale == 1:
            scaledImg = np.copy(normImg)
        else:
            scaledImg = cv2.resize(normImg, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)))
        hsvImg = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2HSV)
        hlsImg = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2HLS)
        nrOfBlocksX = scaledImg.shape[1] // PX_PER_CELL - 1
        nrOfBlocksY = scaledImg.shape[0] // PX_PER_CELL - 1
        windowSizeBlocks = WINDOW_SIZE_PX // PX_PER_CELL - 1
        nrOfStepsX = (nrOfBlocksX - windowSizeBlocks) // CELLS_PER_STEP
        nrOfStepsY = (nrOfBlocksY - windowSizeBlocks) // CELLS_PER_STEP
        hogCh1 = GetHogFeatures(hsvImg[:,:,0], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
        hogCh2 = GetHogFeatures(hsvImg[:,:,1], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
        hogCh3 = GetHogFeatures(hsvImg[:,:,2], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2, isFeatureVector=False)
        for blockNrY in range(nrOfStepsY):
            for blockNrX in range(nrOfStepsX):
                posX = blockNrX * CELLS_PER_STEP
                posY = blockNrY * CELLS_PER_STEP
                # Extract HOG features of the patch
                hogFeatures1 = hogCh1[posY:posY+windowSizeBlocks, posX:posX+windowSizeBlocks].ravel()
                hogFeatures2 = hogCh2[posY:posY+windowSizeBlocks, posX:posX+windowSizeBlocks].ravel()
                hogFeatures3 = hogCh3[posY:posY+windowSizeBlocks, posX:posX+windowSizeBlocks].ravel()
                hogFeatures = np.concatenate((hogFeatures1, hogFeatures2, hogFeatures3))
                windowCoordX = posX * PX_PER_CELL
                windowCoordY = posY * PX_PER_CELL
                # Extract spatial features of the patch
                spatialFeatures = GetSpatialFeatures(hsvImg[windowCoordY:windowCoordY+WINDOW_SIZE_PX, windowCoordX:windowCoordX+WINDOW_SIZE_PX],
                                                     size=(32, 32))
                # Extract color histogram features of the patch
                histFeatures = GetColorHistFeatures(hlsImg[windowCoordY:windowCoordY+WINDOW_SIZE_PX, windowCoordX:windowCoordX+WINDOW_SIZE_PX],
                                                    nrOfBins=32)
                featureList = np.concatenate((hogFeatures, spatialFeatures, histFeatures)).reshape(1, -1)
                featureList = classifier.Normalize(featureList)
                trueWindowCoordX = np.int(windowCoordX * scale)
                trueWindowCoordY = np.int(windowCoordY * scale)
                trueWindowSize = np.int(WINDOW_SIZE_PX * scale)
                if classifier.Predict(featureList) == 1:
                    boundingBoxes.append(((trueWindowCoordX, trueWindowCoordY + rangeY[0]),
                                         (trueWindowCoordX + trueWindowSize, trueWindowCoordY + trueWindowSize + rangeY[0])))
    return boundingBoxes

def GetHeatMap(img, boundingBoxes, threshold=0):
    heat = np.zeros_like(img[:,:,0]).astype(np.uint32)
    for box in boundingBoxes:
        # Add 1 for all pixels inside each box, assuming each box takes the form ((x1, y1), (x2, y2))
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Zero out pixels below the threshold
    heat[heat <= threshold] = 0
    # Visualize the heatmap when displaying    
    return np.clip(heat, 0, 255)
