import os
import argparse
import glob
import cv2
from utils import face_utils
from utils import cv_utils
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions
import random
import json
import time
from tqdm import tqdm

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, expresion):
        img = cv_utils.read_cv2_img(img_path)
        morphed_img = self._img_morph(img, expresion)
        output_name = '%s_out.png' % os.path.basename(img_path)
        self._save_img(morphed_img, output_name)

    def _img_morph(self, img, expresion):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            # face = face_utils.resize_face(img)
            return None

        morphed_face = self._morph_face(face, expresion)

        return morphed_face

    def _morph_face(self, face, expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion/5.0), 0)
        test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['concat']#np.concatenate((imgs['real_img'],imgs['fake_imgs']), axis=1)

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)

def load_expressions(file_path):
    with open(file_path, 'rb') as f:
        return json.load(f)

def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    image_path = opt.input_path

    # # Selected expression
    # data = load_expressions(opt.input_path + "/aus_openface.json")
    # expression = np.array(data["000151"])

    # # Selected random expression
    #expression = np.array(data[random.choice(data.keys())])

    # # Random Masked expression
    expression = np.random.uniform(0, 2, opt.cond_nc)
    mask = np.random.randint(0, 2, opt.cond_nc)
    expression = expression * mask

    # # Custom expression
    # expression = np.array([
    #     1.6,    # 1  Inner Brow Raiser
    #     1.6,    # 2  Outer Brow Raiser
    #     0.0,    # 4  Brow Lowerer
    #     1.2,    # 5  Upper Lid Raiser
    #     0.0,    # 6  Cheek Raiser
    #     0.0,    # 7  Lid Tightener
    #     0.0,    # 9  Nose Wrinkler
    #     0.0,    # 10 Upper Lip Raiser
    #     0.5,    # 12 Lip Corner Puller
    #     0.4,    # 14 Dimpler
    #     0.3,    # 15 Lip Corner Depressor
    #     0.4,    # 17 Chin Raiser
    #     0.0,    # 20 Lip Stretcher
    #     1.2,    # 23 Lip Tightener
    #     0.6,    # 25 Lips part
    #     0.6,    # 26 Jaw Drop
    #     0.1])   # 28 Lip Suck

    print("expression: %s" % (expression))

    filepaths = glob.glob(os.path.join(image_path, '*.jpg'))
    filepaths.sort()

    for image_path in tqdm(filepaths):
        try:
            morph.morph_file(image_path, expression)
        except Exception as e:
            print("Error converting %s" % image_path)

if __name__ == '__main__':
    main()
