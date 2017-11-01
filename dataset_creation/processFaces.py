import imutils
import dlib
import cv2
import numpy as np
import time
import sys, os, errno, argparse



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/data")
fpp = imutils.FacePreProcessor(predictor, desiredFaceWidth=400)





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-I","--images-directory",required=True,help="full path to image-directory")
    ap.add_argument("-O","--output-directory",required=True,help="full path to output-directory")
    args = vars(ap.parse_args())

    #create output directory if not exists
    try:
        os.makedirs(args['output_directory'])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    img_file_names = os.listdir(args["images_directory"])
    os.chdir(args["images_directory"])
    output_dir = "../"+args["output_directory"]+"/"

    for idx,imgname in enumerate(img_file_names):
        print("processing",idx,"  :",imgname)
        image = cv2.imread(imgname,cv2.IMREAD_GRAYSCALE)

        rects = detector(image,1)

        for rect in rects:
            #first, rotate and align the face
            faceAligned,left_eye_center,right_eye_center = fpp.align(image,image,rect)
            #generate a normal distribution of aligned image
            samples = fpp.generate_samples(faceAligned, tuple(left_eye_center), tuple(right_eye_center))

            for i,sample in enumerate(samples):
                cropped_sample = fpp.crop(sample)
                downsampled_sample = cv2.resize(cropped_sample,(32,32))
                normalized_sample = fpp.normalize_intensity(downsampled_sample)
                cv2.imwrite(output_dir + imgname + "_" + str(i), normalized_sample)
