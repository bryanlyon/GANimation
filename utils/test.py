import cv2
import face_utils
import numpy as np

# use any original image as you want
img = cv2.imread('/Users/xyli1905/Projects/Datasets/imgs_178/000009.png')

output, M, angleo, scaleo, chin_percent, chin_xo, chin_yo = face_utils.face_crop_and_align(img)
#output[:,:,1] = output[:,:,0] + 0.01

# for test we use the clipped face rather than the processed face
putback, mask, rotate = face_utils.face_place_back(img, output, angleo, scaleo, chin_percent, chin_xo, chin_yo)

result = np.hstack((img,(1 - mask)*img,rotate,putback))

cv2.imshow('result', result)
cv2.waitKey()