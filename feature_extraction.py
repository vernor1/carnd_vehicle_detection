import cv2
import numpy as np

from skimage.feature import hog


def GetHogFeatures(channel, nrOfOrientations, pxPerCell, cellPerBlk, isVisualization=False, isFeatureVector=True):
    """ Extracts HOG features of the image channel.

    param: channel: 2D array of the image channel
    param: nrOfOrientations: Number of gradient orientations
    param: pxPerCell: Number of pixels per cell
    param: cellPerBlk: Number of cells per block
    param: isVisualization: Indication is visualization is needed
    param: isFeatureVector: Indication if the result needs to be unrolled into a feature vector
    returns: HOG features, [visualization image]
    """
    return hog(channel,
               orientations=nrOfOrientations,
               pixels_per_cell=(pxPerCell, pxPerCell),
               cells_per_block=(cellPerBlk, cellPerBlk),
               transform_sqrt=False,
               visualise=isVisualization,
               feature_vector=isFeatureVector)

def GetSpatialFeatures(img, size=(32, 32), isFeatureVector=True):
    """ Extracts spatial features of the image.

    param: img: Source image
    param: size: Target image size
    param: isFeatureVector: Indication if the result needs to be unrolled into a feature vector
    returns: Spatial features
    """
    resizedImg = cv2.resize(img, size)
    if isFeatureVector:
        return resizedImg.ravel() 
    else:
        return resizedImg

def GetColorHistFeatures(img, nrOfBins=32, isVisualization=False):
    """ Extracts color histogram features of the image.

    param: img: Source image
    param: nrOfBins: NUmber of histogram bins
    param: isVisualization: Indication is visualization is needed
    returns: Color histogram features
    """
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
    """ Extracts image features: HOG, spatial and color histogram.

    param: img: Source image
    returns: Feature vector
    """
    normImg = np.empty_like(img)
    cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Extract HOG features
    ycrcbImg = cv2.cvtColor(normImg, cv2.COLOR_BGR2YCrCb)
    hogFeatures = []
    for channelNr in range(ycrcbImg.shape[2]):
        hogFeatures.append(GetHogFeatures(ycrcbImg[:,:,channelNr], nrOfOrientations=9, pxPerCell=8, cellPerBlk=2))
    hogFeatures = np.ravel(hogFeatures)
    # Extract spatial features
    spatialFeatures = GetSpatialFeatures(ycrcbImg, size=(32, 32))
    # Extract color histogram features
    histFeatures = GetColorHistFeatures(ycrcbImg, nrOfBins=32)
    return np.concatenate((hogFeatures, spatialFeatures, histFeatures))
