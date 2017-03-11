import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calibrate_camera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('camera_cal/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)


    img = cv2.imread("camera_cal/calibration1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibrated!")
    return mtx, dist

#####
#   Calibrate camera and create undistort function
####
mtx, dist = calibrate_camera()
def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

###
#   Performs absolute sobel threshold on input image ing and returns a binary gradient image
##
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

##
#   Returns the image of the s channel in a hls representation of input image "image" that meets the supplied threshold requirements.
##
def color_threshold_for_s_channnel(image, thresh=(0, 255)):
    #color gradient
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary

####
#   Returns the combined sobel and color gradients
###
def grad_col(image):
    gradx = abs_sobel_thresh(image, thresh=(30, 200))
    grady = abs_sobel_thresh(image, orient='y', thresh=(40, 200))
    col_grad = color_threshold_for_s_channnel(image, thresh = (90, 255))

    combined = np.zeros_like(col_grad)
    combined[((gradx == 1) & (grady == 1)) | (col_grad == 1)] = 1
    return combined

def detect_lines(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255.0
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    return leftx, lefty, rightx, righty

width = 1280
height = 720 

offset_y_top = 460
offset_y_bottom = 0
offset_x_top = 556
offset_x_bottom = 0
src = np.float32([[offset_x_top, offset_y_top], [width-offset_x_top, offset_y_top], [width-offset_x_bottom, height-offset_y_bottom], [offset_x_bottom, height-offset_y_bottom]])

dst_width = width - 2* offset_x_bottom
dst_height = height
dst = np.float32([[0,0],[dst_width,0],[dst_width,dst_height],[0,dst_height]])

    
def matrixes():
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

M, Minv = matrixes()

####
##  Create perspective transform
####
def persp_trans(im):
    
    return cv2.warpPerspective(im, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR)

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def addCurvature(image, ploty, left_fit_cr, right_fit_cr):
     
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,'Curvature: ' + str((left_curverad + right_curverad)/2.0),(900,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    return


def addPosition(image, ploty, left_fit, right_fit):
    y_eval = np.max(ploty)
    lane_left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    lane_right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (lane_right_x + lane_left_x) / 2.0
    diff_center = image.shape[1] / 2.0 - lane_center
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,'Position: ' + str(diff_center * xm_per_pix),(900,140), font, 1,(255,255,255),2,cv2.LINE_AA)

def addAnalysis(img, analysis):
    analysis = cv2.resize(analysis,(int(width/4), int(height/4)))
    x_offset=y_offset=50
    analysis = np.dstack((analysis, analysis, analysis))*255
    img[y_offset:y_offset+analysis.shape[0], x_offset:x_offset+analysis.shape[1]] = analysis
    return
    
def process_image(img):
    img = undistort(img)
    gradient_image = grad_col(img)
    binary_warped = persp_trans(gradient_image)
    # Fit a second order polynomial to each

    leftx, lefty, rightx, righty = detect_lines(binary_warped)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #Calculate SI Unit polynomial for curvature calculation
    
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #plt.imshow(result)
    #plt.show()
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    addCurvature(result, ploty, left_fit_cr, right_fit_cr)
    addPosition(result, ploty, left_fit, right_fit)
    addAnalysis(result, binary_warped)
    return result

from moviepy.editor import VideoFileClip

clip1 = VideoFileClip("challenge_video.mp4")
output_video = clip1.fl_image(process_image)
output_video.write_videofile("challenge_output.mp4", audio=False)
