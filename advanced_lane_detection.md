
**Advanced Lane Finding Project**
=================================
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/distorted.jpg "Distorted"
[image1]: ./output_images/undistorted.jpg "Undistorted"
[image2]: ./output_images/middle.jpg "Road distorted"
[image22]: ./output_images/middle_undistorted.jpg "Road undistorted"

[image3]: ./output_images/gradients.jpg "Binary Example"
[image4]: ./output_images/perspective1.jpg "Warp Example"
[image42]: ./output_images/perspective2.jpg "Warp Example2"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/processed.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"

Report
=======

Camera Calibration
------------------


The code for this step is located  in lines 7 through 45 of the file called `standalone.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image0]
![alt text][image1]

Pipeline
--------

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
![alt text][image22]

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 48 through 108 in `standalone.py`).  Here's an example of my output for this step. 

![alt text][image3]


The code for my perspective transform includes a function called `persp_trans()`, which appears in lines 301 through 303 in the file `persp_trans.py`. The `persp_trans()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32([[592, 490], [743, 490], [1126, 710], [308, 710]])

dst = np.float32([[308,0],[1126,0],[1126,720],[308,720]])


```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 592, 490      | 308, 0        | 
| 743, 490      | 1126, 0      |
| 1126, 710     | 1126, 720      |
| 308, 710      | 308, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image42]


To find the lane I used the sliding windows approach to find and follow the lane markings and extract the pixels that make up the lane markings. Using the `np.polyfit()` method I could extract the best 2 nd order polynomial that describes the lane marking. In addition to that I added a constraint that the marking separation should math american lane width standard.
![alt text][image5]


Using the polynomials the curvature could be deduced. I did this in lines 308 through 319 in my code in `standalone.py` in the function `addCurvature()`.


The lane marking was generated from the polynomials and draw to screen. This is implemented in lines 354 to 381 in `standalone.py`in the function `process_image()`  Here is an example of my result on a test image:

![alt text][image6]

---


Here's a [link to my video result](./project_video_output.mp4) 

---

Discussion
----------


There are certainly many areas where this solution can be improved. Fine tuning the gradient and color parameters would be my first area to explore. My solution could be improved by using more colorspace imformation. I focused on the s channel in the hsl colorspace.

Adding constraints (like lane width information) helped and there a more constraints that could be added (Like curvature interdependency). I added some temporal filtering but that could be improved by grading measurements and descriminate against measurements that don't meet certain criterias.

 My solution is most likely sensitive to cars that overtake and disturbs the camera view, weak lane markings, road damage, snow and marks from road repairing.  

