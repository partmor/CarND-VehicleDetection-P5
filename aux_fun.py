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


def extract_spatial_features(img_3ch, color_space='RGB', size=(32, 32)):
    feature_image = np.copy(img_3ch)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features


def extract_hog_features(channel, orientations, pixels_per_cell, cells_per_block, vis=False, feature_vector=True):
    return hog(
        channel, 
        orientations=orientations, 
        pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
        cells_per_block=(cells_per_block, cells_per_block), 
        visualise=vis, 
        feature_vector=feature_vector
    )


def extract_hist_features(img_3ch, nbins=32, bins_range=(0, 256)):
    _, _, _, _, features = color_hist(img_3ch, nbins=nbins, bins_range=bins_range)
    return features