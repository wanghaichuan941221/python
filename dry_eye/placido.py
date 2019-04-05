import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("out1.png")
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

def filter_hsv_image(img,lower_bounds,upper_bounds):
    mask = cv2.inRange(img2,lower_bounds,upper_bounds)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_one_circle(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) > 0:
        biggest_contour = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(biggest_contour)
        return [int(x), int(y),int(r)]
    else:
        return None



lower_bounds,upper_bounds=np.array([0,70,0]),np.array([40,255,100])
mask=filter_hsv_image(img2,lower_bounds,upper_bounds)
[x,y,r]=find_one_circle(mask)

dst=cv2.linearPolar(img,(x,y),5*r,cv2.WARP_FILL_OUTLIERS)
dst = cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)

# lower_bounds1,upper_bounds1 = np.array([0,0,100]),np.array([255,255,255])
# res1=cv2.inRange(dst,lower_bounds1,upper_bounds1)
res1 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

plt.subplot(1,2,1),plt.imshow(dst,cmap="gray")
plt.subplot(1,2,2),plt.imshow(res1,cmap="gray"),plt.show()
