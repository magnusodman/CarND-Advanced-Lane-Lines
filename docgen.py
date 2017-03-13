from standalone import undistort, persp_trans, grad_col3, process_image
import cv2
import matplotlib.pyplot as plt

org = cv2.imread('output_images/last_image.jpg')
"""
im = org.copy()
im = undistort(im)

pts = [(592, 490), (743, 490), (1126, 710), (308, 710)]
cv2.line(im, pts[0], pts[1], (255, 0, 0), thickness=2)
cv2.line(im, pts[1], pts[2], (255, 0, 0), thickness=2)
cv2.line(im, pts[2], pts[3], (255, 0, 0), thickness=2)
cv2.line(im, pts[3], pts[0], (255, 0, 0), thickness=2)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
cv2.imwrite("output_images/perspective1.jpg", im)

#plt.imshow(im)
#plt.show()

im2 = org.copy()
im2 = undistort(im2)
im2 = persp_trans(im2)
pts2 = [(308, 0), (308, 720), (1126, 0), (1126, 720)]
cv2.line(im2, pts2[0], pts2[1], (255, 0, 0), thickness=3)
cv2.line(im2, pts2[2], pts2[3], (255, 0, 0), thickness=3)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
cv2.imwrite("output_images/perspective2.jpg", im2)
#plt.imshow(im2)
#plt.show()


im3 = org.copy()
im3 = undistort(im3)
im3 = grad_col3(im3)
#plt.imshow(im3, cmap="gray")
#plt.show()
cv2.imwrite("output_images/gradients.jpg", im3*255)
"""

im4 = org.copy()
im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
pimg = process_image(im4)
cv2.imwrite("output_images/processed.jpg", pimg)
