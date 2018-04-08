import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import gaussian_filter


# imtensity
def sobel_filter(img):
    h_x=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float64)
    h_y=h_x.transpose()
    gx=signal.convolve2d(img,h_x)
    gy=signal.convolve2d(img,h_y)
    # g=np.sqrt(gx**2+gy**2)
    return gx,gy

def suppression(det,phase):
    gmax=np.zeros(det.shape)
    for i in xrange(gmax.shape[0]):
        for j in xrange(gmax.shape[1]):
            if(phase[i][j]<0):
                phase[i][j]+=360
            if((j+1)<gmax.shape[1] and (j-1)>=0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
                # 0 degree
                if (phase[i][j]>=360-22.5 or phase[i][j]<0+22.5) or (phase[i][j]>=180-22.5 and phase[i][j]<180+22.5):
                    if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
                        gmax[i][j]=det[i][j]
                # 45 degree
                if (phase[i][j]>=45-22.5 and phase[i][j]<45+22.5) or (phase[i][j]>=180+45-22.5 and phase[i][j]<180+45+22.5):
                    if det[i][j] >= det[i-1][j+1] and det[i][j] >= det[i+1][j-1]:
                        gmax[i][j]=det[i][j]
                #90 degree
                if (phase[i][j]>=90-22.5 and phase[i][j]<90+22.5) or (phase[i][j]>=180+90-22.5 and phase[i][j]<180+90+22.5):
                    if det[i][j] >= det[i-1][j] and det[i][j] >= det[i+1][j]:
                        gmax[i][j]=det[i][j]
                #135
                if (phase[i][j]>=135-22.5 and phase[i][j]<135+22.5) or (phase[i][j]>=180+135-22.5 and phase[i][j]<180+135+22.5):
                    if det[i][j] >= det[i-1][j-1] and det[i][j] >= det[i+1][j+1]:
                        gmax[i][j]=det[i][j]
    return gmax




def thresholding(img):
    thres=np.zeros(img.shape)
    nmax = max(img)
    lo,hi=0.1*nmax,0.8*nmax
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i][j] >= hi:
                thres[i][j]=1.0
            elif img[i][j] >= lo:
                thres[i][j]=0.5
    return thres

img = cv2.imread('jpg/apple.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gaussian= gaussian_filter(img_gray,3)
gx,gy= sobel_filter(img_gaussian)
plt.subplot(2,1,1),plt.imshow(img_gray,cmap='gray')
plt.subplot(2,1,2),plt.imshow(np.sqrt(gx**2+gy**2),cmap='gray')
plt.show()
