import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import os
# image = cv2.imread("img1/img100.jpg")

# label = np.loadtxt("img1/img100.txt")

# label_txt = ["turn_left","no_biking","turn_right"]

# fig,ax = plt.subplots(1)
# ax.imshow(image)

# for index in range(label.shape[0]):
#     box = label[index,1:]
#     rect = patches.Rectangle((box[0]*640-box[2]*320,box[1]*480-box[3]*240),box[2]*640,box[3]*480,linewidth=1,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
#     ax.text(box[0]*640-box[2]*320,box[1]*480-box[3]*240,label_txt[index],fontsize=12)
# plt.show()

count = 247
for index1 in range(600):
    image = cv2.imread("img2/img"+str(index1)+".jpg")
    text_path = "img2/img"+str(index1)+".txt"
    if os.path.isfile(text_path):
        label = np.loadtxt(text_path).reshape(-1,5)
        if label.shape[0] != 0 and np.shape(image)!=():
            cv2.imwrite("data/img"+str(count)+".jpg",image)    
            np.savetxt("label/label"+str(count)+".txt",label,fmt="%.4f")
            print("data/img"+str(count)+".jpg")
            count += 1

