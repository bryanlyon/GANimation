import cv2
import numpy as np 
import os
import sys
sys.path.insert(0, 'E:/celebA/GANimation/utils')
import face_utils as face

input_image_path = './sample_dataset/imgs_178/'
output_image_path = './sample_dataset/imgs/'

image_names = os.listdir(input_image_path)

for name in image_names:
    img = cv2.imread(input_image_path+name)
    bb = face.detect_biggest_face(img)
    if bb != None:
        img = img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
        img = face.resize_face(img)
        cv2.imwrite(output_image_path+name, img)