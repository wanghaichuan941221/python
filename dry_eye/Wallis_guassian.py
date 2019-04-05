import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d


def wallis_filter(img1,b,c,mf,sf):
    img = np.zeros((33,63,6,6))
    f = np.zeros((33,63,6,6))
    mg = np.zeros((33,63))
    sg = np.zeros((33,63))
    img2 = np.zeros((198,378))
    for i in range(int(img1.shape[0]/6)):
        for j in range(int(img1.shape[1]/6)):
            img[i][j]=img1[6*i:6*(i+1),6*j:6*(j+1)]
            mg[i][j]=ndimage.mean(img[i][j])
            sg[i][j]=ndimage.variance(img[i][j])
            f[i][j]=(img[i][j]-mg[i][j])*(c*sf/(c*sg[i][j]+(1-c)*sf))+b*mf+(1-b)*mg[i][j]

            for m in range(6):
                for n in range(6):
                    img2[6*i+m][6*j+n]=f[i][j][m][n]
    return img2

img1 = cv2.imread("test.png",0)
img1 = cv2.resize(img1,(378,198))
img1=img1[:198,:378]
mf,sf=120,50
b,c=0.7,0.5

# for m in range(5,8):
#     for n in range(5,8):
#         img2 = wallis_filter(img1,0.1*m,0.1*n,mf,sf)
#         img3 = cv2.GaussianBlur(img2,(9,9),0)
#         img3 = img2.astype(np.uint8)
#         plt.subplot(3,3,3*(m-5)+n-4),plt.imshow(3*(img3-img1),cmap="gray")
#
# plt.show()

img2 = wallis_filter(img1,b,c,mf,sf)

img3 = cv2.GaussianBlur(img2,(9,9),0)
# img3 = img2.astype(np.uint8)

img4 = 2*(img3-img1)
img4 = cv2.GaussianBlur(img4,(11,11),0)
img4 = img4.astype(np.uint8)

# plt.hist(img4.ravel(),256,[0,256]),plt.show()
res1 = cv2.adaptiveThreshold(img4,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
k1=np.ones((5,5),np.uint8)
res2 = cv2.dilate(res1,k1,1)
# res2=255-res2

k2 = np.ones((2,2),np.uint8)
res3=cv2.erode(res2,k2,1)
# res3 = 255-res3

plt.subplot(2,2,1),plt.imshow(img1,cmap="gray")
plt.subplot(2,2,2),plt.imshow(img2,cmap="gray")
plt.subplot(2,2,3),plt.imshow(res2,cmap="gray")
plt.subplot(2,2,4),plt.imshow(res3,cmap="gray")
plt.show()
