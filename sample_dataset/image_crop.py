import cv2
import numpy as np 
import os

input_image_path = './imgs_178/'
output_image_path = './imgs_128/'
file_list = os.listdir(input_image_path)

for img_name in file_list:
    name = img_name.split('.')
    img = cv2.imread(input_image_path+img_name)
    img = img[20:198, 0:]
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    img = cv2.resize(img, (128,128), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_image_path + name[0]+'.jpg', img)