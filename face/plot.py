import cv2
import matplotlib.pyplot as plt
import numpy as np

img_d = np.zeros((74,67*11))
for num in range(1,12):
    img = cv2.imread('img'+str(num)+'.JPG')
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img,(67,74))
    img_d[:,(67*num-67):(67*num)]=img
    plt.subplot(1,11,num),plt.imshow(img)

plt.show()
