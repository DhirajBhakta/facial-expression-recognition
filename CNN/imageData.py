import dlib
import numpy as np
import cv2
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
from dataset_creation.imutils import FacePreProcessor, rect_to_bb

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/data")
fpp = FacePreProcessor(predictor, desiredFaceWidth=400)

#pre-process image without sample generation
#this function is not used in training nor testing
#used ONLY while predicting

class ImageData:
    def __init__(self, imagefilename):
        self.image = cv2.imread(imagefilename)
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.rects = detector(self.gray,1)
        self.preprocessed_faces = self.preprocess()

    def preprocess(self):
        faces = []
        for rect in self.rects:
            faceAligned = fpp.align(self.gray,self.gray,rect)[0]
            cropped_sample = fpp.crop(faceAligned)
            downsampled_sample = cv2.resize(cropped_sample,(32,32))
            normalized_sample = fpp.normalize_intensity(downsampled_sample)
            faces.append(normalized_sample)
        return np.array(faces)

    def showResults(self, predictions):
        for rect, prediction in zip(self.rects, predictions):
            (x,y,w,h) = rect_to_bb(rect)
            cv2.rectangle(self.image, (x,y), (x+w,y+h), (255,0,0), 2)
            dy=int(h//20)
            for expression in prediction.keys():
                expr = expression +" :"+str(prediction[expression]*100)+"%"
                cv2.putText(self.image, expr ,(x,y),2,dy*0.1,(0,0,255))
                y+=5*dy
                break
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)

        cv2.imshow("output",self.image)
        cv2.waitKey(0)
