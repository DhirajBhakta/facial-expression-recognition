from .face_utils import FACIAL_LANDMARKS_IDXS
from .face_utils import shape_to_np
import numpy as np
# from scipy.ndimage.filters import generic_filter
import cv2
import math
from random import gauss

class FacePreProcessor:
    def __init__(self, predictor, desiredLeftEye=(0.35,0.35), desiredFaceWidth=256, desiredFaceHeight=None):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if(desiredFaceHeight is None):
            self.desiredFaceHeight = desiredFaceWidth
        self.initializeInvariants()

    def initializeInvariants(self):
        desiredRightEyeX = 1 - self.desiredLeftEye[0]
        self.desiredDistance = (desiredRightEyeX - self.desiredLeftEye[0]) * self.desiredFaceWidth

        self.tx = self.desiredFaceWidth *0.5
        self.ty = self.desiredFaceHeight *self.desiredLeftEye[1]

        horizontal_crop_len = 1.5* self.desiredDistance
        vertical_crop_len = 2.25* self.desiredDistance
        self.crop_pos_x_left = int(self.tx - horizontal_crop_len/2)
        self.crop_pos_x_right= int(self.crop_pos_x_left + horizontal_crop_len)
        self.crop_pos_y_top = int(self.ty - vertical_crop_len/3)
        self.crop_pos_y_bottom= int(self.crop_pos_y_top + vertical_crop_len)


    def align(self, image, gray, rect):
        shape = self.predictor(gray,rect)
        shape = shape_to_np(shape)

        (left_eye_start, left_eye_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_eye_start, right_eye_end) = FACIAL_LANDMARKS_IDXS["right_eye"]

        left_eye_pts = shape[left_eye_start:left_eye_end]
        right_eye_pts = shape[right_eye_start:right_eye_end]

        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")

        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy,dx))-180

        dist = np.sqrt((dx**2)+(dy**2))
        scale = self.desiredDistance/dist

        eyesCenter = ((left_eye_center[0]+right_eye_center[0])//2 , (left_eye_center[1]+right_eye_center[1])//2)

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        M[0,2] += (self.tx - eyesCenter[0])
        M[1,2] += (self.ty - eyesCenter[1])

        (w,h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(gray,M,(w,h),flags=cv2.INTER_CUBIC)

        #return aligned image
        #also return eye centers , which aid in sample generation.
        return output,left_eye_center, right_eye_center



    def crop(self, aligned_image):

        return aligned_image[self.crop_pos_y_top:self.crop_pos_y_bottom, self.crop_pos_x_left:self.crop_pos_x_right]

    def generate_samples(self,aligned_image, left_eye_center, right_eye_center):
        mean = 0
        stddev = 3 # 3 degrees
        lefteye_random_angles = [gauss(mean,stddev) for i in range(4)]
        righteye_random_angles = [gauss(mean,stddev) for i in range(4)]

        (w,h) = (self.desiredFaceWidth, self.desiredFaceHeight)

        samples = []
        for angle in lefteye_random_angles:
            M = cv2.getRotationMatrix2D(left_eye_center,angle,1)
            output = cv2.warpAffine(aligned_image,M,(w,h),flags=cv2.INTER_CUBIC)
            samples.append(output)

        for angle in righteye_random_angles:
            M = cv2.getRotationMatrix2D(right_eye_center,angle,1)
            output = cv2.warpAffine(aligned_image,M,(w,h),flags=cv2.INTER_CUBIC)
            samples.append(output)

        return samples




    def normalize_intensity(self, image):
        return cv2.equalizeHist(image)
