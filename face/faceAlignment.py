from imutils import face_utils
import numpy as np
import cv2
import dlib
import imutils
import argparse

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()-x
    h = rect.bottom()-y
    return (x,y,w,h)

def shape_to_np(shape,dtype="int"):
    coords = np.zeros((68,2),dtype=dtype)
    for i in range(0,68):
        coords[i]=(shape.part(i).x,shape.part(i).y)
    return coords

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.share_predictor(".")

img = cv2.imread('img1.JPEG')
img = imutils.resize(img,width=500)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

rects = detector(gray,1)

for (i,rect) in enumerate(rects):
    shape = predictor(gray,rect)
    shape = face_utils.shape_to_np(shape)
    (x,y,w,h)=face_utils.rect_to_bb(rect)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(img,"Face #{}".format(i+1),(x-10,y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x,y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("output",img)
cv2.Waitkey(0)
