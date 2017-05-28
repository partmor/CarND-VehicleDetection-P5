# **CarND: Vehicle Detection and Tracking**  [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
[//]: # (Image References)

[two_samples]: ./output_images/two_samples.png
[pos_hog_sample]: ./output_images/pos_hog_sample.png
[pos_3ch]: ./output_images/pos_3ch.png
[car_hist]: ./output_images/car_hist.png
[scale_1_75]: ./output_images/scale_1_75.png
[scale_2_75]: ./output_images/scale_2_75.png
[heat_map]: ./output_images/heat_map.png

The goal of this project is to build a software pipeline to **detect and track surrounding vehicles** in a video stream generated by a front-facing camera mounted on a car.

At its core, this pipeline uses a machine learning classifier model, to classify video-frame patches as *vehicle* or *non-vehicle*.

The project is separated into two well-differentiated blocks:
+ **Training stage**: feature extraction pipeline for the training dataset, and model training.
+ **Prediction stage**: feature extraction pipeline for the video stream frames.

It is possible to create a unique feature extraction pipeline that suits both training and prediction stages. However, this is not optimal since this would require a computationally expensive sliding window search implementation to predict on the new input video frames. This will be discussed later in more detail. 

### Project structure

The project workflow is built in the notebook `p5_solution.ipynb`, linked [here](p5_solution.ipynb). To improve the readability of the notebook and make it less bulkier, the basic core methods and other helpers have been extracted to the `aux_fun.py` script. These functions are cited as `aux_fun.method_name()`, whereas those that are defined in the actual project notebook are simply cited as `method_name()`.

## Training set construction

To train the classifier, a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) has been used in this project.

The merged database contains labeled 64 x 64 RGB images of *vehicles* and *non-vehicles*, in a near-50% proportion. 

| Positive        | Negative   |  Total |
|:-------------:|:-------------:| :-------------:|
| 8792     | 8968       |  17760  |

Here is an example of each class:

![two_samples]

Three types of features are extracted from the database images in this project:
+ Histogram of Oriented Gradients (**HOG**), using `aux_fun.extract_hog_features()`.
+ **Color histogram** features, using `aux_fun.extract_hist_features()`.
+ **Spatial features**, i.e. resized image raw pixel values, using `aux_fun.extract_spatial_features()`.

For the initial car example, the HOG features extracted from the Y channel are:

![pos_hog_sample]

The pixel values in YCrCb space:

![pos_3ch]

And finally the histograms for each of the YCrCb channels: 

![car_hist]

To be consumable by the classifier model, the feature arrays must be flattened and concatenated together into a one-dimensional vector. 

For a single image, the total feature vector extraction is encapsulated in `extract_single_img_features()`. This processed is looped for each image in the database with the help of `extract_features_from_img_file_list()`. Finally `build_dataset()` merges and shuffles all the data into the final training set, and provides the label vector populated with zeros and ones. 

## Model training pipeline

The model used for the classication task in this project is a **Linear Support Vector Machine Classifier**, in particular [`sklearn.svm.LinearSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). This election is supported on the fact it is a rather simple method that is known to perform well in image classification taks, while being faster than higher order kernel SVMs and complex tree based models. It is important for the model to be fast at predicting, since it must be able to process a video stream.

The training set construction involved the joint usage of features that present disparity of magnitudes: gradient values, raw pixel values, and histogram counts. It is therefore necessary to perform a scaling of the features so one type does not dominate over the rest during the training process. 

Feature scaling is performed using [`sklearn.preprocessing.StandardScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). A great advantage of using scikit-learn's API, is that preprocessing steps can be enclosed together with the model itself in a *pipeline object*, [`sklearn.pipeline.Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), that exposes the same methods than the classifier, but without having to worry about explicitly calling the preprocessing steps everytime the model is used, improving readability and robustness.

The pipeline object is built with the [`sklearn.pipeline.make_pipeline()`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) method:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

scaled_clf = make_pipeline(
    StandardScaler(),
    LinearSVC()
)
```

The effect of the hyperparameters on the performance of the model is assessed using a 3-fold stratified cross validation, to ensure the 50-50 class balance is preserved in each fold.

---

**NOTE**: The **GTI** image database presents multiple images for each real world entity, since the images have been extracted from near-to-consecutive video frames. If this fact is not taken into account, the obtained CV scores will be overly optimistic due to **leakage**: it is possible to find an image of the same entity in both training and validation folds at each step. This could be easily resolved using time series data or entity-labelled data together with a **Group K-Fold method**, `sklearn.model_selection.GroupKFold`.

This extra information is not available. I tried defining clusters based on template matching with `cv2.matchTemplate()`, where images with high resemblance would be grouped together, but was not robust enough. Further approaches were not tried due to lack of time.

---

## Prediction pipeline

The peculiarity of the prediction stage, is that the inputs are video frames, that must be hovered by a sliding window that extracts features for the model to decide if the window's region is a vehicle or not.

The objective is to perform a **sliding window search**, with a given value of overlap in both directions of space, and extract all the features (recall HOG, color histograms, and spatial features) for each window providing a prediction. However, this approach is overly computationally expensive, since features are extracted multiple times in a given region due to window overlapping.

A more efficient solution is to **extract the HOG features only once**, on the whole image, and subsample the resulting array along with the moving windows. This improved methodology is encapsulated in `find_cars()`.

In this implementation, **window scale** is controlled with the `scale` parameter. The scale parameter resizes the image so that the sliding windows have an effective size of `64*scale`.

**Overlapping** is controlled with the relation between the total number of HOG cells in a 64-px window and `cells_per_step_overlap`. An example: regardless of the scale, `hog_pixels_per_cell=16` involves 4 x 4  cells in a 64 x 64 window. Hence, `cells_per_step_overlap=1` yields 75 % overlap, `cells_per_step_overlap=2` 50 %, and so on.

The following image shows the detections with scale 1.0 windows and 75 % overlap:

![scale_1_75]

whereas the image bellow presents detections with a 2.0 scale (128 x 128 windows) and 75 % overlapping:

![scale_2_75]

As seen, in a given frame there are many overlapping positive detections surrounding the vehicles, but there can also be false positives. **False positives** tend to appear in a sparse manner, so their presence in a frame can be resolved by building tight bounding boxes to the *hot regions* in a **thresholded heatmap**. Continuing with the example case from above:

![heat_map]

The detection pipeline is encapsulated in `video_pipeline()`.
