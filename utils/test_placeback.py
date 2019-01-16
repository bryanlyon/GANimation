import cv2
import face_utils
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='input path to test image')
parser.add_argument('--test_img_path', type=str, 
					default='/Users/xyli1905/Projects/Datasets/imgs_178/000009.png', 
					help='path to the test image')

arg = parser.parse_args()

# use any original image as you want
img = cv2.imread(arg.test_img_path)

output, face_origin_pos = face_utils.face_crop_and_align(img)
#output[:,:,1] = output[:,:,0] + 0.01

# for test we use the clipped face rather than the processed face
# note: input/output method changes when need use edge-connect later:
#	    for input need to add maskA = maskA,
#		for output it changes to placeback, new_maskA
placeback, mask, rotate = face_utils.face_place_back(img, output, face_origin_pos)
#placeback, new_maskA = face_utils.face_place_back(img, output, face_origin_pos, 
#												  maskA = np.array([[1.,0.2,0.3],[0.6,0.7,0.8]]))

result = np.hstack((img,(1 - mask)*img,rotate,placeback))

cv2.imshow('result', result)
while True:
	key = cv2.waitKey()
	if key == 27:
		break;