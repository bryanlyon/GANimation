import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import re
import pickle
import json
import random
import cv2
import face_recognition

parser = argparse.ArgumentParser()
parser.add_argument('-ia', '--input_aus_filesdir', type=str, help='Dir with imgs aus files')
parser.add_argument('-ii', '--input_images_filesdir', type=str, help='Dir with imgs image files')
parser.add_argument('-op', '--output_path', type=str, help='Output path')
args = parser.parse_args()

def get_data(filepaths, qualitycutoff):
    data = dict()
    for filepath in tqdm(filepaths):
        content = np.loadtxt(filepath, delimiter=', ', skiprows=1)
        if content.ndim == 1:
            if content[1] > qualitycutoff:
                data[os.path.basename(filepath[:-4])] = content[(-17*2-1):-18]
        else:
            if content[0][1] > qualitycutoff:
                data[os.path.basename(filepath[:-4])] = content[0][(-17*2-1):-18]

    return data

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open(name + '.json', 'wb') as f:
        data2 = {}
        for x,y in data.items():
            data2[x] = y.tolist()
        json.dump(data2, f)


def get_random_split(data, set1ratio, set2ratio):
    datalen = len(data)
    ratioamount = datalen / (set1ratio + set2ratio)
    set1len = ratioamount * set1ratio
    print("Datalen: %s Trainlen: %s Testlen: %s" % (datalen, set1len, datalen-set1len))
    random.shuffle(data)
    print(len(data))
    set1=data[:set1len]
    set2=data[set1len:]
    set1.sort()
    set2.sort()
    return set1, set2

def crop_face_with_bb(img, bb):
    '''
    Crop face in image given bb
    :param img: cv::mat HxWx3
    :param bb: 4 (<x,y,w,h>)
    :return: HxWx3
    '''
    x, y, w, h = bb
    return cv2.resize(img[y:y+h, x:x+w, :], (128, 128))

def find_face(img):
    bbs = face_recognition.face_locations(img)
    if len(bbs) > 0:
        y, right, bottom, x = bbs[0]
        bb = x, y, (right - x), (bottom - y)
        face = crop_face_with_bb(img, bb)
        return face
    else:
        return None

def main():
    filepaths = glob.glob(os.path.join(args.input_aus_filesdir, '*.csv'))
    filepaths.sort()

    file_path = os.path.join(args.output_path, "aus_openface.pkl")
    print("File path: %s" % file_path)
    with open(file_path, 'rb') as f:
        data=pickle.load(f)

    # create aus file
    # data = get_data(filepaths, .85)

    print( "Found data: %s" % len(data))
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(args.output_path + "/imgs/")
    save_dict(data, os.path.join(args.output_path, "aus_openface"))

    img_list=[]
    bad_list=[]

    for item in tqdm(data.keys()):
        if os.path.exists(args.output_path + "/imgs/%s.bmp" % item):
            img_list.append(item)
            pass
        else:
            image_path = args.input_images_filesdir + "%s.jpg" % item
            img = cv2.imread(image_path, -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_img = find_face(img)

            if face_img is not None:
                output_path = args.output_path + "/imgs/%s.bmp" % item
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(output_path, face_img)
                img_list.append(item)
            else:
                bad_list.append(item)

    trainset, testset = get_random_split(img_list,10,1)

    print( "Train set: %s" % len(trainset))
    with open(os.path.join(args.output_path, "train_ids.csv"), 'w') as f:
        for item in trainset:
            f.write(item + ".bmp\n")

    print( "Test set: %s" % len(testset))
    with open(os.path.join(args.output_path, "test_ids.csv"), 'w') as f:
        for item in testset:
            f.write(item + ".bmp\n")

    print( "Bad set: %s" % len(bad_list))
    with open(os.path.join(args.output_path, "bad_ids.csv"), 'w') as f:
        for item in bad_list:
            f.write(item + ".bmp\n")

if __name__ == '__main__':
    main()
