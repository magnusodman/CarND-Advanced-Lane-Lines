from standalone import grad_col3
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
import numpy as np
import cv2

def hls(image):
    #color gradient
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    return H, L, S


def thresh(image, thresh):
    binary = np.zeros_like(image)
    image[(S < thresh[0]) | (S > thresh[1])] = 0
    
    return image

image = cv2.imread('middle.jpg')

H, L, S = hls(image.copy())

plt.imshow(thresh(S, (50,255)), cmap="gray")
plt.show()

    