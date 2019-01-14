import face_recognition
import cv2
import numpy as np
import skimage
import skimage.transform
import warnings

def detect_faces(img):
    '''
    Detect faces in image
    :param img: cv::mat HxWx3 RGB
    :return: yield 4 <x,y,w,h>
    '''
    # detect faces
    bbs = face_recognition.face_locations(img)

    for y, right, bottom, x in bbs:
        # Scale back up face bb
        yield x, y, (right - x), (bottom - y)

def detect_biggest_face(img):
    '''
    Detect biggest face in image
    :param img: cv::mat HxWx3 RGB
    :return: 4 <x,y,w,h>
    '''
    # detect faces
    bbs = face_recognition.face_locations(img)

    max_area = float('-inf')
    max_area_i = 0
    for i, (y, right, bottom, x) in enumerate(bbs):
        area = (right - x) * (bottom - y)
        if max_area < area:
            max_area = area
            max_area_i = i

    if max_area != float('-inf'):
        y, right, bottom, x = bbs[max_area_i]
        return x, y, (right - x), (bottom - y)

    return None

def crop_face_with_bb(img, bb):
    '''
    Crop face in image given bb
    :param img: cv::mat HxWx3
    :param bb: 4 (<x,y,w,h>)
    :return: HxWx3
    '''
    x, y, w, h = bb
    return img[y:y+h, x:x+w, :]

def place_face(img, face, bb):
    x, y, w, h = bb
    face = resize_face(face, size=(w, h))
    img[y:y+h, x:x+w] = face
    return img

def resize_face(face_img, size=(128, 128)):
    '''
    Resize face to a given size
    :param face_img: cv::mat HxWx3
    :param size: new H and W (size x size). 128 by default.
    :return: cv::mat size x size x 3
    '''
    return cv2.resize(face_img, size)

def detect_landmarks(face_img):
    landmakrs = face_recognition.face_landmarks(face_img)
    return landmakrs[0] if len(landmakrs) > 0 else None

def face_crop_and_align(face_img, chin_percent=0.95):
    landMark = detect_landmarks(face_img)

    left_eye = np.array(landMark['left_eyebrow'])
    left_eye_mean = np.mean(left_eye, axis= 0)

    right_eye = np.array(landMark['right_eyebrow'])
    right_eye_mean = np.mean(right_eye, axis = 0)

    eye_mean = (left_eye_mean + right_eye_mean)/2


    dY = right_eye_mean[1] - left_eye_mean[1]
    dX = right_eye_mean[0] - left_eye_mean[0]
    angle = np.degrees(np.arctan2(dY, dX)) 

    # mouth = np.vstack((np.array(landMark['bottom_lip']), np.array(landMark['top_lip'])))
    chin = np.array(landMark['chin'])
    #print(chin)

    index = np.argmax(chin[:,1])
    chin_max = chin[index]


    dist = np.sqrt((eye_mean[0] - chin_max[0])**2 + (eye_mean[1] - chin_max[1])**2)
    padding = chin_percent*128-dist
    #print(chin_percent, padding, dist)

    if padding < 0:
        padding = 0
    
    dist = dist + padding

    scale = chin_percent*128/dist
    #print(scale)

    M = cv2.getRotationMatrix2D((chin_max[0], chin_max[1]), angle, scale)

    M[0,2] += 128*0.5 - chin_max[0]
    M[1,2] += 128*chin_percent - chin_max[1]
    output = cv2.warpAffine(face_img, M, (128, 128), flags=cv2.INTER_CUBIC)
    return output, M, angle, scale, chin_percent, chin_max[0], chin_max[1]

def face_place_back(image, face_img, angleo, scaleo, chin_percent, chin_xo, chin_yo):
    '''
    image     --- the original image
    face_img  --- processed sub img (128X128) that only has the face
    angle     --- angle used when clipping face for processing
    scale     --- scale used when clipping face for processing
    chin_percent --- predefined portion
    chin_xo, chin_yo --- position of chin in original image
    '''

    # if we use a mask of the same size as clipped figure, 
    # due to the precision problem in rotation will cause discontinuity
    # instead, we build a slightly smaller mask for putting back, 
    # avoinding discontinuity
    margin = 3

    masko = np.ones((128 - margin,128 - margin,3), dtype = np.uint8)
    masko[0:margin,:,:] = 0
    masko[:,0:margin,:] = 0

    # find affine matrix M
    chin_x = 128 * 0.5
    chin_y = 128 * chin_percent
    angle  = -angleo
    scale  = 1./scaleo

    M = cv2.getRotationMatrix2D((chin_x, chin_y), angle, scale)
    M[0,2] += chin_xo - chin_x
    M[1,2] += chin_yo - chin_y

    # putback editted figure
    h = image.shape[1]
    w = image.shape[0]
    rotate = cv2.warpAffine(face_img, M, (h, w), flags=cv2.INTER_CUBIC)
    mask = cv2.warpAffine(masko, M, (h, w), flags=cv2.INTER_CUBIC)

    putback = (1 - mask) * image + mask * rotate

    return putback, mask, rotate
