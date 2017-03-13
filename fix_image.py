import cv2
from standalone import undistort
img = cv2.imread('middle.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imwrite("output_images/middle.jpg", img)

undistoreted = undistort(img)

cv2.imwrite("output_images/middle_undistorted.jpg", undistoreted)