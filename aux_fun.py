import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import hog
from sklearn.utils import shuffle

#############
#  HELPERS  #
#############


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def slide_window_uniform(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(image, labels):
    img = np.copy(image)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def plot_3_channels(img, ch_names=('Ch. 1', 'Ch. 2', 'Ch. 3')):
    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,5))
    ax1.imshow(
        img[:,:,0],
        cmap='gray'
    )
    ax1.axis('off')
    ax1.set_title(ch_names[0])
    ax2.imshow(
        img[:,:,1],
        cmap='gray'
    )
    ax2.set_title(ch_names[1])
    ax2.axis('off')
    ax3.imshow(
        img[:,:,2],
        cmap='gray'
    )
    ax3.axis('off')
    ax3.set_title(ch_names[2])



########################
#  FEATURE EXTRACTION  #
########################


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


def extract_spatial_features(img, size=(32, 32)):
    feature_image = np.copy(img)             
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


def extract_single_img_features(img_rgb, 
                                use_hog, hog_cspace, channels_for_hog,
                                hog_orientations, hog_pixels_per_cell, hog_cells_per_block,
                                use_hist, hist_cspace, hist_nbins,
                                use_spatial, spatial_cspace, spatial_resize
                               ):
    
    img_spatial = img_rgb if spatial_cspace=='RGB' else convert_color(img_rgb, 'RGB2%s' % spatial_cspace)
    img_hist = img_rgb if hist_cspace=='RGB' else convert_color(img_rgb, 'RGB2%s' % hist_cspace)
    img_hog = img_rgb if hog_cspace=='RGB' else convert_color(img_rgb, 'RGB2%s' % hog_cspace)
    
    spatial_features = extract_spatial_features(img_spatial, size=(spatial_resize,)*2)
    hist_features = extract_hist_features(img_hist, nbins=hist_nbins)
    hog_features_list = list()
    for ch in channels_for_hog:
        hog_features_list.append(
            extract_hog_features(img_hog[:,:,ch], hog_orientations, hog_pixels_per_cell, hog_cells_per_block)
        )
    hog_features = np.concatenate(hog_features_list)
    
    features_list = list()
    if use_spatial:
        features_list.append(spatial_features)
    if use_hist:
        features_list.append(hist_features)
    if use_hog:
        features_list.append(hog_features)
    
    return np.concatenate(features_list)


def extract_features_from_img_file_list(img_fname_list,
                                        use_hog, hog_cspace, channels_for_hog,
                                        hog_orientations, hog_pixels_per_cell, hog_cells_per_block,
                                        use_hist, hist_cspace, hist_nbins,
                                        use_spatial, spatial_cspace, spatial_resize
                                       ):
    feature_list = list()
    for img_fname in img_fname_list:
        img_features = list()
        img_rgb = io.imread(img_fname)
        feature_list.append(
            extract_single_img_features(img_rgb, 
                                        use_hog, hog_cspace, channels_for_hog,
                                        hog_orientations, hog_pixels_per_cell, hog_cells_per_block,
                                        use_hist, hist_cspace, hist_nbins,
                                        use_spatial, spatial_cspace, spatial_resize
                                       )
        )
    return np.vstack(feature_list)


def build_dataset(positive_fname_list, negative_fname_list, 
                    use_hog, hog_cspace, channels_for_hog,
                    hog_orientations, hog_pixels_per_cell, hog_cells_per_block,
                    use_hist, hist_cspace, hist_nbins,
                    use_spatial, spatial_cspace, spatial_resize                 
                 ):
    
    extract = lambda x: extract_features_from_img_file_list(
        x, 
        use_hog, hog_cspace, channels_for_hog,
        hog_orientations, hog_pixels_per_cell, hog_cells_per_block,
        use_hist, hist_cspace, hist_nbins,
        use_spatial, spatial_cspace, spatial_resize   
    )          
    
    X_0 = extract(negative_fname_list)
    y_0 = np.zeros(X_0.shape[0])
    X_1 = extract(positive_fname_list)
    y_1 = np.ones(X_1.shape[0])
    return shuffle(np.vstack([X_0, X_1]), np.concatenate([y_0, y_1]))


########################
#  PREDICTION PIPELINE #
########################


def find_cars(img, ystart, ystop, scale, cells_per_step_overlap, clf_pl, 
              orient, pix_per_cell, cell_per_block, 
              spatial_size, 
              hist_bins,
              use_hog=True,
              hog_cspace='LUV',
              use_spatial=True,
              spatial_cspace='RGB',
              use_hist=True,
              hist_cspace='RGB',
              channels_for_hog=[0,1,2],
              vis=False
             ):
    
    draw_img = np.copy(img)
    img_tosearch = img[ystart:ystop,:,:]
    
    if scale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Define blocks and steps as above
    nxblocks = (img_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (img_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = cells_per_step_overlap  # Instead of overlap, define how many cells to step (2 default)
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Color space transformation for each feature extraction method
    ctrans_tosearch_hog = img_tosearch if hog_cspace=='RGB' else convert_color(img_tosearch, 'RGB2%s' % hog_cspace)
    ctrans_tosearch_spatial = img_tosearch if spatial_cspace=='RGB' else convert_color(img_tosearch, 'RGB2%s' % spatial_cspace)
    ctrans_tosearch_hist = img_tosearch if hist_cspace=='RGB' else convert_color(img_tosearch, 'RGB2%s' % hist_cspace)
    
    # Compute individual channel HOG features for the entire image
    hog_features_per_channel = list()
    for i in channels_for_hog:
        hog_i = extract_hog_features(
            ctrans_tosearch_hog[:,:,i], 
            orient, pix_per_cell, cell_per_block, feature_vector=False
        )
        hog_features_per_channel.append(hog_i)
    
    hot_window_list = list()
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_subsampled_features_list = list()
            for hog_ch_feats in hog_features_per_channel:
                hog_ch_feats_subsample = hog_ch_feats[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_subsampled_features_list.append(hog_ch_feats_subsample)
            hog_features = np.concatenate(hog_subsampled_features_list)

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg_spatial = cv2.resize(ctrans_tosearch_spatial[ytop:ytop+window, xleft:xleft+window], (64,64))
            subimg_hist = cv2.resize(ctrans_tosearch_hist[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get color features
            spatial_features = extract_spatial_features(subimg_spatial, size=(spatial_size,)*2)
            hist_features = extract_hist_features(subimg_hist, nbins=hist_bins)

            # Scale features and make a prediction
            feature_list = list()
            if use_spatial:
                feature_list.append(spatial_features)
            if use_hist:
                feature_list.append(hist_features)
            if use_hog:
                feature_list.append(hog_features)
            test_features = np.concatenate(feature_list)[np.newaxis]  
            test_prediction = clf_pl.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart))
                hot_window_list.append(bbox)
                if vis:
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),3) 
    if vis:
        return hot_window_list, draw_img
    return hot_window_list