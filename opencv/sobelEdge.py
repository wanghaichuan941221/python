import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def sobel_filter(img,ksize):
    if ksize==3:
        h_x=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float64)
        h_y=h_x.transpose()
        gx=signal.convolve2d(img,h_x)
        gy=signal.convolve2d(img,h_y)
        # g=np.sqrt(gx**2+gy**2)
        return gx,gy
    elif ksize==5:
        h_x=np.array([[-1,-2,0,2,1],[-4,-8,0,8,4],[-6,-12,0,12,6],[-4,-8,0,8,4],[-1,-2,0,2,1]])
        h_y=h_x.transpose()
        gx=signal.convolve2d(img,h_x)
        gy=signal.convolve2d(img,h_y)
        return gx,gy



src_apple = cv2.imread('jpg/apple.jpg')
src_gray_apple = cv2.cvtColor(src_apple,cv2.COLOR_BGR2GRAY)

gx,gy=sobel_filter(src_gray_apple,5)

dst_x = cv2.Sobel(src_gray_apple,cv2.CV_64F,1,0,ksize=3)
dst_y = cv2.Sobel(src_gray_apple,cv2.CV_64F,0,1,ksize=3)
plt.subplot(2,2,1),plt.imshow(dst_x,cmap='gray')
plt.subplot(2,2,2),plt.imshow(dst_y,cmap='gray')
plt.subplot(2,2,3),plt.imshow(gx,cmap='gray')
plt.subplot(2,2,4),plt.imshow(gy,cmap='gray')

plt.show()
