import cv2
import numpy as np 
import os
import sys
sys.path.insert(0, 'E:/celebA/GANimation/utils')
import face_utils as face

input_image_path = './sample_dataset/imgs_178/'
output_image_path = './sample_dataset/imgs_align/'

image_names = os.listdir(input_image_path)
error = open('error.txt', 'w')
for name in image_names:
    img = cv2.imread(input_image_path+name)
    name = name.split('.')[0]
    try:
        output, M, angle = face.face_crop_and_align(img)
    except:
        try:
            bb = face.detect_biggest_face(img)
            output = img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
        except:
            print('cannot find face')
            continue
    if output.shape[0] != 128 or output.shape[1] != 128:
        output = cv2.resize(output, (128,128))
        # print('resize : ',name)
    cv2.imwrite(output_image_path+name+'.jpg', output)

error.close()