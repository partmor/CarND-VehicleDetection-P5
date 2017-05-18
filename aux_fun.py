import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def color_hist(img, nbins=32, bins_range=(0, 256), plot=False, ch_names=list('RGB')):
    # Compute the histogram of the RGB channels separately
    ch1 = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ch2 = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    ch3 = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = ch1[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((ch1[0], ch2[0], ch3[0]))
    # Visualize
    if plot:
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.bar(bin_centers, ch1[0])
        plt.xlim(bins_range)
        plt.title('%s Histogram' % ch_names[0])
        plt.subplot(132)
        plt.bar(bin_centers, ch2[0])
        plt.xlim(bins_range)
        plt.title('%s Histogram' % ch_names[1])
        plt.subplot(133)
        plt.bar(bin_centers, ch3[0])
        plt.xlim(bins_range)
        plt.title('%s Histogram' % ch_names[2])
        fig.tight_layout()
    # Return the individual histograms, bin_centers and feature vector
    return ch1, ch2, ch3, bin_centers, hist_features


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features


def get_hog_features(channel, orientations, pixels_per_cell, cells_per_block, vis=False, feature_vector=True):
    return hog(
        channel, 
        orientations=orientations, 
        pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
        cells_per_block=(cells_per_block, cells_per_block), 
        visualise=vis, 
        feature_vector=feature_vector
    )