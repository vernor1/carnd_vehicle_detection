import cv2
import numpy as np

from skimage.feature import hog

def GetHogFeatures(channel, nrOfOrientations, pxPerCell, cellPerBlk, isVisualization=False, isFeatureVector=True):
    return hog(channel,
               orientations=nrOfOrientations,
               pixels_per_cell=(pxPerCell, pxPerCell),
               cells_per_block=(cellPerBlk, cellPerBlk),
               transform_sqrt=False,
               visualise=isVisualization,
               feature_vector=isFeatureVector)

def GetSpatialFeatures(img, size=(32, 32), isFeatureVector=True):
    resizedImg = cv2.resize(img, size)
    if isFeatureVector:
        return resizedImg.ravel() 
    else:
        return resizedImg

def GetColorHistFeatures(img, nrOfBins=32, isVisualization=False):
    # Compute the histogram of the RGB channels separately
    histB = np.histogram(img[:,:,0], bins=nrOfBins)
    histG = np.histogram(img[:,:,1], bins=nrOfBins)
    histR = np.histogram(img[:,:,2], bins=nrOfBins)
    # Concatenate the histograms into a single feature vector
    histFeatures = np.concatenate((histR[0], histG[0], histB[0]))
    if not isVisualization:
        return histFeatures
    else:
        # Generating bin centers
        binEdges = histR[1]
        binCenters = (binEdges[1:]  + binEdges[0:len(binEdges)-1]) / 2
        # Return the individual histograms, bin centers and feature vector
        return histFeatures, histR, histG, histB, binCenters

def ExtractFeatures(img):
    normImg = np.empty_like(img)
    cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Extract HOG features
    hsvImg = cv2.cvtColor(normImg, cv2.COLOR_BGR2HSV)
    hogFeatures = []
    for channelNr in range(hsvImg.shape[2]):
        hogFeatures.append(GetHogFeatures(hsvImg[:,:,channelNr], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2))
    hogFeatures = np.ravel(hogFeatures)
    # Extract spatial features
    spatialFeatures = GetSpatialFeatures(hsvImg, size=(32, 32))
    # Extract color histogram features
    histFeatures = GetColorHistFeatures(cv2.cvtColor(normImg, cv2.COLOR_BGR2HLS), nrOfBins=32)
    return np.concatenate((hogFeatures, spatialFeatures, histFeatures))
