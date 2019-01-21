import face_recognition
import cv2
import numpy as np
import skimage
import skimage.transform
import warnings
from .face_utils import detect_landmarks

def face_crop_and_align(origin_img, chin_percent=0.95, eye_center_percent=0.2):
    # detect landmarks on the original image
    landmarks = detect_landmarks(origin_img)

    # find the mean center of left and right eyes
    left_eye = np.array(landmarks['left_eyebrow'])
    left_eye_mean = np.mean(left_eye, axis = 0)

    right_eye = np.array(landmarks['right_eyebrow'])
    right_eye_mean = np.mean(right_eye, axis = 0)

    eye_center = (left_eye_mean + right_eye_mean)/2.0

    # determine the angle for rotating face
    # eye_difference[0]: difference in x-axis
    # eye_difference[1]: difference in y-axis
    eye_difference = right_eye_mean - left_eye_mean
    angle = np.degrees(np.arctan2(eye_difference[1], eye_difference[0]))

    # load chin landmarks and find the lowest point of the chin
    chin = np.array(landmarks['chin'])
    index = np.argmax(chin[:,1])
    chin_max = chin[index]

    # fine distence between eye_center and lowest point of chin
    # and determine the proper scale factor of the face, making it fit our model
    dist = np.linalg.norm(eye_center - chin_max)
    scale = 128 * (chin_percent - eye_center_percent) / dist

    # generate affine matrix for face cropping
    M = cv2.getRotationMatrix2D((chin_max[0], chin_max[1]), angle, scale)
    M[0,2] += 128 * 0.5 - chin_max[0]
    M[1,2] += 128 * chin_percent - chin_max[1]

    # crop the original image, producing a standarized face_img
    face_img = cv2.warpAffine(origin_img, M, (128, 128), flags=cv2.INTER_CUBIC)

    # wrap up for information of original face position
    face_origin_pos = {"angleo": angle,
                       "scaleo": scale,
                       "chin_percent": chin_percent,
                       "chin_xo": chin_max[0],
                       "chin_yo": chin_max[1],
                      }

    return face_img, face_origin_pos

def face_place_back(origin_img, face_img, face_origin_pos, **kwargs):
    '''
    origin_img --- the original image
    face_img   --- processed sub img (128X128) that focus on the face
    face_origin_pos --- dict that contains position info of face in oringal image
    kwargs --- in csae if need edge-connect to process the image further,
               to use kwargs: maskA = maskA
    maskA --- mask A in the GANimation
    '''

    # if we use a mask of the same size as clipped figure, 
    # due to the precision problem in rotation will cause discontinuity
    # instead, we build a slightly smaller mask for putting back, 
    # avoinding discontinuity
    margin = 3

    masko = np.ones((128 - margin,128 - margin,3), dtype = np.uint8)
    masko[0:margin,:,:] = 0
    masko[:,0:margin,:] = 0

    # find the inverse affine matrix M
    chin_x = 128 * 0.5
    chin_y = 128 * face_origin_pos['chin_percent']
    angle  = - face_origin_pos['angleo']
    scale  = 1./face_origin_pos['scaleo']

    M = cv2.getRotationMatrix2D((chin_x, chin_y), angle, scale)
    M[0,2] += face_origin_pos['chin_xo'] - chin_x
    M[1,2] += face_origin_pos['chin_yo'] - chin_y

    # putback processed figure
    h = origin_img.shape[1]
    w = origin_img.shape[0]
    rotate = cv2.warpAffine(face_img, M, (h, w), flags=cv2.INTER_CUBIC)
    mask = cv2.warpAffine(masko, M, (h, w), flags=cv2.INTER_CUBIC)

    processed_img = (1 - mask) * origin_img + mask * rotate

    # if need edge-connect, construct the mask for further processing
    # new mask is generated based on mask A and a threshold
    if kwargs != {}:
        if kwargs['maskA_test']:
            maskA = 1.0 - kwargs['maskA']
            new_maskA = cv2.warpAffine(maskA, M, (h, w), flags=cv2.INTER_CUBIC)

            return processed_img, mask, rotate, 1.0 - new_maskA
        else:
            maskA = kwargs['maskA']
            threshold = 0.65

            maskA[maskA <= threshold] = 0.
            maskA[maskA > threshold] = 1.

            new_maskA = cv2.warpAffine(maskA, M, (h, w), flags=cv2.INTER_CUBIC)
            new_maskA = new_maskA.astype(np.uint8)

            return processed_img, new_maskA

    #return processed_img, mask, rotate
    return processed_img