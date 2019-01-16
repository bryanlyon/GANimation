import argparse
import os
import glob
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import cv2
import numpy as np
from networks import generatorFoward
import utils.util as util
import utils.face_utils as face
import pickle
import time
import random

class feedFoward:
    def __init__(self, path):
        self._model = generatorFoward.generatorFoward(conv_dim=64, c_dim=17, repeat_num=6)
        self._model.load_state_dict(torch.load(path, map_location='cpu'))
        self._model.eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def Foward(self, face, desired_expression):
        face = torch.unsqueeze(self._transform(face), 0).float()
        desired_expression = torch.unsqueeze(torch.from_numpy(desired_expression/5.0), 0).float()
        start = time.clock()
        color, mask = self._model.forward(face, desired_expression)
        end = time.clock()
        print('forward time : %2.5f (s)' % (end - start))
        # torch.onnx.export(self._model,(face, desired_expression), "alexnet.onnx", verbose=True)
        masked_result = mask * face + (1.0 - mask) * color

        img = self.convertToimg(masked_result)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        maskA = mask.detach().numpy()[0]
        maskA = np.transpose(maskA, (1,2,0))

        return img, maskA
    
    def convertToimg(self, tensor):
        tensor = tensor.cpu().float()
        img = tensor.detach().numpy()[0]
        img = img*0.5+0.5
        img = np.transpose(img, (1, 2, 0))
        return img*254.0

def find_epoch(model_path, load_epoch_num):
    if os.path.exists(model_path):
        if load_epoch_num == -1:
            epoch_num = 0
            for file in os.listdir(model_path):
                if file.startswith("net_epoch_"):
                    epoch_num = max(epoch_num, int(file.split('_')[2]))
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

def main():
    parser = argparse.ArgumentParser(description='input path to original image')
    parser.add_argument('--img_path', type=str, 
                        default='/Users/xyli1905/Projects/Datasets/imgs_178/000009.png', 
                        help='path to the test image')
    parser.add_argument('--model_path', type=str, default='./checkpoints/model_align/', 
                        help='path to the pretrained model')
    parser.add_argument('--load_epoch', type=int, default=-1, help='specify the model to be loaded')

    arg = parser.parse_args()

    # use any original image as you want and clip it
    img_raw = cv2.imread(arg.img_path)
    #print(np.shape(img_raw))
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    real_face, face_origin_pos = face.face_crop_and_align(img)

    # load pretrained GANimation model and run
    #path = './checkpoints/original_model/net_epoch_30_id_G.pth'
    epoch_num = find_epoch(arg.model_path, arg.load_epoch)
    load_filename = 'net_epoch_%s_id_G.pth' % (epoch_num)
    path = os.path.join(arg.model_path, load_filename)
    convertor = feedFoward(path)

    # set expression
    expressions = np.ndarray((5,17), dtype = np.float)
    a = np.array([0.25, 0.11, 0.2 , 0.16, 1.92, 1.03, 0.3 , 2.15, 2.88, 1.61, 0.03, 0.09, 0.16, 0.11, 2.25, 0.37, 0.05], dtype = np.float)
    #b = np.array([0.69, 0, 0.01 , 0, 0.19, 1.02, 0 , 0, 1.25, 1.41, 0, 0, 0.59, 0.05, 0.9, 0, 0.33], dtype = np.float)/10
    #c = (b - a)/4
    #for i in range(5):
    #   expressions[i] = c * i + a

    # run model for expression 'a' only for now
    processed_face, maskA = convertor.Foward(real_face, a)
    new_img, mask, rotate, new_maskA = face.face_place_back(img_raw, processed_face, face_origin_pos, 
                                                            mask_test=True, maskA = maskA)

    # handle maskA to image for dispalying
    maskA_t = np.expand_dims(new_maskA, axis=2)
    maskA_t = (maskA_t*254.0).astype(np.uint8)
    maskA_t = np.repeat(maskA_t, 3, axis=-1)

    result = img_raw
    result = np.hstack((result, rotate, maskA_t, new_img))
    #print(np.shape(processed_face), np.shape(maskA), maskA.dtype)

    cv2.imshow('result', result/254.0)

    while True:
        key = cv2.waitKey()
        if key == 27:
            break;

if __name__ == '__main__':
    main()