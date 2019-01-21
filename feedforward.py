import argparse
import os
import glob
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import cv2
import numpy as np
from networks import smile_forward
import utils.util as util
import utils.smile_utils as face
import pickle
import calendar
import time
import random

class feedForward:
    def __init__(self, pathG, pathD):
        # load pre-trained generator here
        self._modelG = smile_forward.generatorFoward(conv_dim=64, c_dim=17, repeat_num=6)
        self._modelG.load_state_dict(torch.load(pathG, map_location='cpu'))
        self._modelG.eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])
        #load pre-trained discriminator here
        self._modelD = smile_forward.Discriminator(image_size=128, conv_dim=64, c_dim=17, repeat_num=6)
        self._modelD.load_state_dict(torch.load(pathD, map_location='cpu'))
        self._modelD.eval()

    def Foward(self, face, desired_expression):
        # transform face by normalizing 
        face = torch.unsqueeze(self._transform(face), 0).float()
        desired_expression = torch.unsqueeze(torch.from_numpy(desired_expression/5.0), 0).float() # set up desired expression

        color, mask = self._modelG.forward(face, desired_expression)
        # calculate result
        masked_result = mask * face + (1.0 - mask) * color

        img = self.convertToimg(masked_result)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)# convert to OpenCV color BGR

        maskA = mask.detach().numpy()[0]
        maskA = np.transpose(maskA, (1,2,0))

        return img, maskA

    def FindAU(self, face):
        # transform face by normalizing 
        face = torch.unsqueeze(self._transform(face), 0).float()
        out_real, out_aux = self._modelD.forward(face)
        return out_real, out_aux
    
    def convertToimg(self, tensor):
        tensor = tensor.cpu().float()
        img = tensor.detach().numpy()[0]
        img = img*0.5+0.5 # reverse transform
        img = np.transpose(img, (1, 2, 0))
        return img*254.0

def find_epoch(model_path, load_epoch_num):
    if os.path.exists(model_path):
        if load_epoch_num == -1:
            epoch_num = 0
            for file in os.listdir(model_path):
                if file.startswith("net_epoch_"):
                    epoch_num = max(epoch_num, int(file.split('_')[2]))
            print("load epoch %d" % epoch_num)
        else:
            found = False
            for file in os.listdir(model_path):
                if file.startswith("net_epoch_"):
                    found = int(file.split('_')[2]) == load_epoch_num
                    if found: break
            assert found, 'Model for epoch %i not found' % load_epoch_num
            epoch_num = load_epoch_num
    else:
        assert load_epoch_num < 1, 'Model for epoch %i not found' % load_epoch_num
        epoch_num = 0

    return epoch_num

def load_target_AU():
    '''
    use predefined target AUs for big and small smile expressions;
    row_1 if for big smile, row_2 is for small smile
    '''
    # target_AU = np.array(
    # [[0.2 , 0.12, 0.1 , 0.24, 0.05, 0.12, 0.08, 0.11, 0.19, 0.22, 0.1 , 0.09, 0.15, 0.07, 0.11, 0.16, 0.03],
    #  [0.4 , 0.1 , 0.28, 0.45, 0.46, 0.32, 0.36, 1.61, 0.72, 0.45, 0.34, 0.28, 0.29, 0.09, 0.34, 0.3 , 0.07],
    #  [0.51, 0.37, 0.17, 1.53, 0.09, 0.16, 0.1 , 0.21, 0.21, 0.21, 0.19, 0.15, 0.2 , 0.09, 0.23, 0.29, 0.04],
    #  [1.02, 0.34, 2.  , 0.54, 0.32, 0.66, 0.25, 0.6 , 0.38, 0.56, 0.33, 0.33, 0.3 , 0.13, 0.28, 0.36, 0.09],
    #  [0.27, 0.17, 0.08, 0.26, 0.51, 0.38, 0.12, 0.57, 1.46, 0.56, 0.04, 0.04, 0.16, 0.06, 1.22, 0.44, 0.03],
    #  [0.27, 0.2 , 0.14, 0.26, 0.07, 0.19, 0.07, 0.23, 0.2 , 0.23, 0.16, 0.1 , 0.2 , 0.06, 0.43, 1.19, 0.05],
    #  [1.46, 0.76, 0.5 , 0.79, 1.15, 0.95, 0.2 , 1.48, 1.94, 1.04, 0.11, 0.13, 0.3 , 0.09, 1.79, 0.53, 0.06],
    #  [0.67, 0.35, 0.34, 0.51, 0.47, 0.7 , 0.26, 0.84, 0.54, 0.5 , 0.4 , 0.26, 0.33, 0.14, 1.14, 2.12, 0.09],
    #  [1.52, 0.8 , 0.24, 0.41, 0.15, 0.21, 0.07, 0.23, 0.3 , 0.32, 0.23, 0.19, 0.32, 0.09, 0.25, 0.29, 0.05],
    #  [2.07, 1.32, 0.49, 1.72, 0.4 , 0.38, 0.11, 0.56, 0.56, 0.65, 0.33, 0.54, 0.42, 0.2 , 0.39, 0.39, 0.06],
    #  [0.38, 0.13, 0.27, 0.32, 0.4 , 1.5 , 0.27, 0.25, 0.45, 0.31, 0.23, 0.15, 0.37, 0.12, 0.28, 0.37, 0.04],
    #  [0.92, 0.34, 0.75, 0.66, 1.47, 2.64, 0.69, 0.98, 0.84, 0.61, 0.56, 0.72, 0.91, 0.28, 0.47, 0.34, 0.05],
    #  [0.23, 0.07, 0.18, 0.16, 1.04, 1.66, 0.28, 0.68, 1.64, 0.96, 0.07, 0.13, 0.31, 0.16, 0.82, 0.34, 0.04],
    #  [0.17, 0.1 , 0.1 , 0.17, 1.11, 0.48, 0.15, 1.35, 2.3 , 1.19, 0.02, 0.05, 0.13, 0.08, 1.7 , 0.3 , 0.03],
    #  [0.26, 0.13, 0.13, 0.22, 0.3 , 0.26, 0.12, 0.4 , 0.96, 1.37, 0.07, 0.27, 0.24, 0.21, 0.21, 0.25, 0.04],
    #  [0.65, 0.29, 0.33, 0.49, 0.67, 0.44, 0.27, 0.8 , 0.56, 0.74, 0.62, 1.73, 0.6 , 0.44, 0.11, 0.38, 0.08],
    #  [0.25, 0.11, 0.2 , 0.16, 1.92, 1.03, 0.3 , 2.15, 2.88, 1.61, 0.03, 0.09, 0.16, 0.11, 2.25, 0.37, 0.05],
    #  [0.34, 0.1 , 0.48, 0.23, 1.44, 0.89, 0.68, 1.84, 1.63, 1.7 , 0.22, 0.63, 0.34, 0.32, 0.53, 0.32, 0.07],
    #  [0.19, 0.06, 0.16, 0.14, 1.59, 1.9 , 0.27, 1.32, 2.38, 0.92, 0.04, 0.03, 0.2 , 0.08, 1.98, 0.45, 0.04],
    #  [0.32, 0.06, 0.49, 0.17, 2.37, 2.65, 0.71, 2.16, 2.65, 1.4 , 0.08, 0.07, 0.32, 0.13, 2.29, 0.55, 0.05]], dtype = np.float)
    target_AU = np.array(
    [[0.25, 0.11, 0.2 , 0.16, 1.92, 1.03, 0.3 , 2.15, 2.88, 1.61, 0.03, 0.09, 0.16, 0.11, 2.25, 0.37, 0.05],
     [0.26, 0.13, 0.13, 0.22, 0.3 , 0.26, 0.12, 0.4 , 0.96, 1.37, 0.07, 0.27, 0.24, 0.21, 0.21, 0.25, 0.04]], dtype = np.float)
    return target_AU

def img_processing(img_raw, convertor, expression_num = 5, test=True):

    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    real_face, face_origin_pos = face.face_crop_and_align(img)

    # find original AU of input image using discrinator of GANimation, for test use only
    out_real, out_aux = convertor.FindAU(real_face) # out_aux is the AU value from D 
    original_AU = out_aux.data.numpy()

    # define the target AU, for test use only
    target_AU = load_target_AU()

    # work for 2 predefined expressions: big smile and small smile
    # set expression

    row_num = np.shape(target_AU)[0]

    expressions = np.ndarray((expression_num, row_num, 17), dtype = np.float)

    # interpolate between original_AU and the target_AU, for test only
    o_AU = np.repeat([original_AU], row_num, axis = 0)
    assert np.shape(o_AU) == np.shape(target_AU), "shape not match"

    c = (o_AU - target_AU)/(expression_num - 1)
    for i in range(expression_num):
        expressions[i] = c * i + target_AU

    # run model for above expressions
    if test :
        result = np.zeros((1, img_raw.shape[1] * (expression_num + 1), 3))
        for j in range(row_num):
            current_result = img_raw
            for i in range(expression_num):
                processed_face, _ = convertor.Foward(real_face, expressions[i, j])
                new_img = face.face_place_back(img_raw, processed_face, face_origin_pos)
                current_result = np.hstack((current_result, new_img))
            result = np.vstack((result, current_result))

        return result
    else:
        big_smile = []
        small_smile = []
        for j in range(row_num):
            for i in range(expression_num):
                processed_face, _ = convertor.Foward(real_face, expressions[i, j])
                new_img = face.face_place_back(img_raw, processed_face, face_origin_pos)
                big_smile.append(new_img) if j == 0 else small_smile.append(new_img)

        dict_smile_face = {"big_smile": big_smile, "small_smile": small_smile}

        return dict_smile_face

class input_options:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--img_path', type=str, 
                                  default='/Users/xyli1905/Projects/Datasets/CelebA_178/000009.png', 
                                  help='path to the test image')
        self._parser.add_argument('--model_path', type=str, 
                                  default='/Users/xyli1905/Projects/Image_Processing/GANimation/checkpoints/model_align', 
                                  help='path to the pretrained model')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='specify the model to be loaded')

        self.initialized = True

    def parse(self):
        if not self._initialized :
            self.initialize()
        self._args = self._parser.parse_args()

        return self._args

# =========================================================== #
# below, the main(), is a demo for how to use the feedforward #
# =========================================================== #
def main():
    # load input parameters
    arg = input_options().parse()

    image_name = arg.img_path.split('.')[-2]
    print(image_name)
    image_name = image_name.split('/')[-1]

    # use any original image as you want and clip it
    img_raw = cv2.imread(arg.img_path)
    img_raw = cv2.resize(img_raw,(0,0), fx=1, fy=1)

    # load pretrained GANimation model and run
    epoch_num = find_epoch(arg.model_path, arg.load_epoch)
    load_filename_generator = 'net_epoch_%s_id_G.pth' % (epoch_num)
    load_filename_discriminator = 'net_epoch_%s_id_D.pth' % (epoch_num)
    pathG = os.path.join(arg.model_path, load_filename_generator)
    pathD = os.path.join(arg.model_path, load_filename_discriminator)

    convertor = feedForward(pathG, pathD)

    if_test = False # for test only
    expression_num = 5
    #dict_smile_face = img_processing(img, convertor, original_AU, target_AU)
    result = img_processing(img_raw, convertor, expression_num, test = if_test)

    if if_test :
        timestamp = calendar.timegm(time.gmtime())
        image_out_name = "./results/processedface_"+image_name+"-"+str(timestamp)+".jpg"
        cv2.imwrite(image_out_name, result)
        print("Processed image saved as %s" % image_out_name)
        #cv2.imshow('result', result/254.0)
        #cv2.waitKey()
    else:
        # dump individual processed image
        # for i in range(expression_num):
        #     image_tmp_1 = result["big_smile"][i]
        #     image_tmp_2 = result["small_smile"][i]

        #     timestamp = calendar.timegm(time.gmtime())
        #     image_name_big = "./results/art/individual/big_smile-"+str(i)+"-"+str(timestamp)+".jpg"
        #     image_name_small = "./results/art/individual/small_smile-"+str(i)+"-"+str(timestamp)+".jpg"

        #     cv2.imwrite(image_name_big, image_tmp_1)
        #     cv2.imwrite(image_name_small, image_tmp_2)
        timestamp = calendar.timegm(time.gmtime())
        img_hstack = np.hstack((img_raw, result["big_smile"][0], result["big_smile"][1], result["big_smile"][2]))
        image_out_name = "./results/hstack_bigsmile_"+image_name+"-"+str(timestamp)+".jpg"
        cv2.imwrite(image_out_name, img_hstack)
        print("Processed image saved as %s" % image_out_name)
        # cv2.imshow('big_smile', image_tmp_1/254.0)
        # cv2.imshow('small_smile', image_tmp_2/254.0)
        # cv2.waitKey()

if __name__ == '__main__':
    main()
#