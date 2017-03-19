##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

[car_and_notcar]: ./output_images/car_and_notcar.png

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

[hog]: ./output_images/hog.png

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of `skimage.hog()` parameters and color spaces based on course material. I eventually settled on these following parameters since they gave me the best results: `color_space: YCrCb`, `orientations = 9`, `pix_per_cell = 8`, and `cell_per_block = 2`. Then the test accuracy was 0.9896.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used a combination of HOG features, histograms of color(number of hist bins = 32) and spatial binning(spatial_size = (32, 32)). The data was split into training and test sets with the test size at 20% of the total data. The resulting feature vector length was 8460. Using the YCrCb color resulted in better performance. Then the test accuracy was 99%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My code was based on the course material, which requires computing hog features only once for the whole image. I used two window scales. The first one is 1.0 which I searched with from y=400 to 656. The second one is 1.5.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

[test-window1]: ./output_images/test-window1.png
[test-window2]: ./output_images/test-window2.png
[test-window3]: ./output_images/test-window3.png
[test-window4]: ./output_images/test-window4.png
[test-window5]: ./output_images/test-window5.png
[test-window6]: ./output_images/test-window6.png
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are some example images:

[test1]: ./output_images/test1.png
[test2]: ./output_images/test2.png
[test3]: ./output_images/test3.png
[test4]: ./output_images/test4.png
[test5]: ./output_images/test5.png
[test6]: ./output_images/test6.png
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My major issue was that there are too many false positives such as an oncoming car, other car and shadow, so I tried to minimize them. Hyper parameter choosing especially scale was effective for reducing the false positives in my video. However, I had high processing time to get the output video using these hyper parameters. I think I need to improve my pipeline and process_img functions to reduce the processing time.

If I make my pipeline more robust for conditions such as rain, snow and fog, I could use a different classifier, increase the training data or combine the current classifier with a convolutional neural network. I think that Deep Learning approach is effective, so I will try it.
