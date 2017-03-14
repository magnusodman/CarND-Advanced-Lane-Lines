import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pandas

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
    col_grad = color_threshold_for_s_channnel(image, thresh = (50, 255))

    combined = np.zeros_like(col_grad)
    combined[((gradx == 1) & (grady == 1)) | (col_grad == 1)] = 1
    return combined

def grad_col2(image):
    gradx = abs_sobel_thresh(image, thresh=(10, 255))
    grady = abs_sobel_thresh(image, orient='y', thresh=(10, 100))
    col_grad = color_threshold_for_s_channnel(image, thresh = (50, 150))

    combined = np.zeros_like(col_grad)
    combined[((gradx == 1) & (grady == 1)) | (col_grad == 1)] = 1
    return combined

def grad_col3(image, x_thresh=(10, 255), y_thresh=(10, 100), col_thresh=(40, 255)):
    gradx = abs_sobel_thresh(image, thresh=x_thresh)
    grady = abs_sobel_thresh(image, orient='y', thresh=y_thresh)
    col_grad = color_threshold_for_s_channnel(image, thresh = col_thresh)

    combined = np.zeros_like(col_grad)
    combined[((gradx == 1) & (grady == 1)) | (col_grad == 1)] = 1
    return combined

class LaneFollower:
    leftx_base = None
    rightx_base = None

    def detect_lines(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255.0
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)

        meta_data = {}

        if self.leftx_base == None:
            self.leftx_base = np.argmax(histogram[:midpoint])
        if self.rightx_base == None:
            self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint


        temp_leftx_base = np.argmax(histogram[:midpoint])        
        temp_rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        lane_distance =  abs(temp_leftx_base - temp_rightx_base)
        if lane_distance < 730 and lane_distance > 680:
            if abs(temp_leftx_base - self.leftx_base) < 40 and abs(temp_rightx_base - self.rightx_base) < 40:
                self.leftx_base = temp_leftx_base
                self.rightx_base = temp_rightx_base
        
        meta_data["leftx_base"] = self.leftx_base
        meta_data["rightx_base"] = self.rightx_base
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Set the width of the windows +/- margin

        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = margin / 2
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
                #Make sure not drifting apart from other line
                distance = np.abs(np.int(np.mean(nonzerox[good_left_inds])) - rightx_current)
                if distance < 1000:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                distance = np.abs(leftx_current- np.int(np.mean(nonzerox[good_right_inds])))
                if distance < 1000:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        meta_data["out_img"] = out_img
        return leftx, lefty, rightx, righty, meta_data


    polyHistory = []
    polyFilterLeft = []
    polyFilterRight = []
    polyFilterLeftCR = []
    polyFilterRightCR = []
    
    """
    Returns lane plynomials for SI and pixel units 
    """
    def lanePolynomials(self, binary_warped):
        left_fit, right_fit, left_fit_cr, right_fit_cr, meta_data = self.lanePolynomialsCurrent(binary_warped)
        
        self.polyFilterLeft.insert(0,left_fit)
        self.polyFilterRight.insert(0,right_fit)
        self.polyFilterLeftCR.insert(0,left_fit_cr)
        self.polyFilterRightCR.insert(0,right_fit_cr)

        filteredLeft = np.mean(self.polyFilterLeft, axis=0)
        filteredRight =np.mean(self.polyFilterRight, axis=0)
        filteredLeftCR = np.mean(self.polyFilterLeftCR, axis=0)
        filteredRightCR =np.mean(self.polyFilterRightCR, axis=0)


        if len(self.polyFilterLeft) > 10:
            self.polyFilterLeft.pop()
        if len(self.polyFilterRight) > 10:
            self.polyFilterRight.pop()
        if len(self.polyFilterLeftCR) > 10:
            self.polyFilterLeftCR.pop()
        if len(self.polyFilterRightCR) > 10:
            self.polyFilterRightCR.pop()

        return filteredLeft, filteredRight, filteredLeftCR, filteredRightCR, meta_data
 
    last_left_fit = None
    last_right_fit = None
    last_left_fit_cr = None
    last_right_fit_cr = None
    def lanePolynomialsCurrent(self, binary_warped):
        # Fit a second order polynomial to each

        leftx, lefty, rightx, righty, meta_data = self.detect_lines(binary_warped)
    
        if len(leftx) == 0:
            left_fit = self.last_left_fit
            left_fit_cr = self.last_left_fit_cr
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.last_left_fit = left_fit
            #Calculate SI Unit polynomial for curvature calculation
            left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
            self.last_left_fit_cr = left_fit_cr
        
        if len(rightx) == 0:
            right_fit = self.last_right_fit
            right_fit_cr = self.last_right_fit_cr
        else:
            right_fit = np.polyfit(righty, rightx, 2)
            self.last_right_fit = right_fit
            #Calculate SI Unit polynomial for curvature calculation
            right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
            self.last_right_fit_cr = right_fit_cr

        meta_data["left_fit"] = left_fit
        meta_data["right_fit"] = right_fit
        
        self.polyHistory.append((left_fit, right_fit))

        return left_fit, right_fit, left_fit_cr, right_fit_cr, meta_data


width = 1280
height = 720 

src = np.float32([[592, 470], [743, 470], [1126, 710], [308, 710]])
dst_width = 1126 - 308
dst_height = 720



dst_width = 1280
dst_height = 720

dst = np.float32([[308,0],[1126,0],[1126,720],[308,720]])

    
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
    curvature = max([left_curverad, right_curverad])
    
    cv2.putText(image,'Curvature: {0:.2f}'.format(curvature) ,(900,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    return


def addPosition(image, ploty, left_fit, right_fit):
    y_eval = np.max(ploty)
    lane_left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    lane_right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (lane_right_x + lane_left_x) / 2.0
    diff_center = image.shape[1] / 2.0 - lane_center
    diff_m= diff_center * xm_per_pix
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,'Position: {0:.2f}'.format(diff_m),(900,140), font, 1,(255,255,255),2,cv2.LINE_AA)

def addAnalysis(img, meta_data, grad_image, binary_warped):
    analysis = meta_data["out_img"]
    analysis = cv2.resize(analysis,(int(width/4), int(height/4)))
    analysis2 = cv2.resize(grad_image,(int(width/4), int(height/4)))
    analysis2 = np.dstack((analysis2, analysis2, analysis2))*255

    x_offset=y_offset=50
    x_offset2 = x_offset + analysis.shape[1]

    #analysis = np.dstack((analysis, analysis, analysis))*255
    
    img[y_offset:y_offset+analysis.shape[0], x_offset:x_offset+analysis.shape[1]] = analysis
    img[y_offset:y_offset+analysis2.shape[0], x_offset2:x_offset2+analysis2.shape[1]] = analysis2

    #meta_data["left_fit"]
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img,'Left fit: ' + str(meta_data["left_fit"]),(900,300), font, 1,(255,255,255),2,cv2.LINE_AA)
    return

lane_follower = LaneFollower()

def process_image(img):
    undistorted = undistort(img)
    gradient_image = grad_col2(undistorted)
    binary_warped = persp_trans(gradient_image)
    
    
    left_fit, right_fit, left_fit_cr, right_fit_cr, meta_data = lane_follower.lanePolynomials(binary_warped)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    if left_fit[0]*right_fit[0] < 0:
        pass
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
    if False:
        addAnalysis(result, meta_data, gradient_image, binary_warped)
    return result

if __name__=="__main__":
    if True:
        from moviepy.editor import VideoFileClip
        clip1 = VideoFileClip("project_video.mp4")
        #clip1 = VideoFileClip("challenge_video.mp4")
        output_video = clip1.fl_image(process_image)
        output_video.write_videofile("project_video_output.mp4", audio=False)
        #output_video.write_videofile("challenge_video_output.mp4", audio=False)
    else:
        img = cv2.imread('last_image.jpg')
        pimg = process_image(img)
        plt.imshow(pimg)
        plt.show()


