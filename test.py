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

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, desired_expression, origin_expression):
        img = cv_utils.read_cv2_img(img_path)
        morphed_img = self._img_morph(img, desired_expression, origin_expression)
        output_name = '%s_out.png' % os.path.basename(img_path)
        self._save_img(morphed_img, output_name)

    def _img_morph(self, img, desired_expression, origin_expression):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)

        morphed_face = self._morph_face(img, desired_expression, origin_expression)

        return morphed_face

    def _morph_face(self, face, desired_expression, origin_expression):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        desired_expression = torch.unsqueeze(torch.from_numpy(desired_expression/10.0), 0)
        origin_expression = torch.unsqueeze(torch.from_numpy(origin_expression/5.0), 0)
        test_batch = {'real_img': face, 'real_cond': origin_expression, 'desired_cond': desired_expression, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['concat']

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('test.jpg', img)
        cv2.imshow('result', img)
        cv2.waitKey()


def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    image_path = opt.input_path
    # expression = np.random.uniform(0, 1, opt.cond_nc)
    desired_expression = np.array([0, 0, 0.16, 0, 1.75, 0.58, 0, 1.32, 2.82, 1.13, 0, 0, 0.08, 0, 1.93, 0, 0], dtype = np.float)
    origin_expression = np.array([0, 0, 0.46, 0, 0, 0, 0.03, 0, 0, 0.59, 0.02, 0.22, 0, 0.21, 0, 0, 0], dtype = np.float)

    morph.morph_file(image_path, desired_expression, origin_expression)



if __name__ == '__main__':
    main()
