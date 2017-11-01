# takes in arguments
#   --the Directory containing Images (rotated, cropped and stddev reproduced)
#   --and the labels

import os
import cv2
import argparse
from array import array


ap = argparse.ArgumentParser()
ap.add_argument("-I","--images-directory",required=True,help="full path to image-directory")
ap.add_argument("-L","--labels-file",required=True,help="full path to labels-file")
args = vars(ap.parse_args())




dict_of_labels = {}

#open and load labels file
with open(args["labels_file"],"r") as labelsfile:
    for line in labelsfile:
        (image_file_name, label) = line.split()
        dict_of_labels[image_file_name] = int(label) -1
        # why minus 1? : because LABLES.txt contains labels from 1-7. we need 0-6




impackfile = open("im.pck","wb")
lblpackfile = open("lbl.pck","wb")
image_file_names = os.listdir(args["images_directory"])
total_images = len(image_file_names)

#first int data in the file represents total number of images in the directory
impackfile.write(total_images.to_bytes(4,'big'))

#next , get dimensions of an image in this Directory
os.chdir(args["images_directory"])
tmp_im = cv2.imread(image_file_names[0],cv2.IMREAD_GRAYSCALE)
(height, width) = tmp_im.shape

#second int data in the file represents the height of each image in directory.
impackfile.write(height.to_bytes(4,'big'))
#third int data in the file represents the width of each image in directory.
impackfile.write(width.to_bytes(4,'big'))

for i,imagefilename in enumerate(image_file_names):
    index = imagefilename[0:8]
    if index in dict_of_labels.keys():
        print("packaging image ",i," :",imagefilename)
        label = dict_of_labels[index]
        lblpackfile.write(label.to_bytes(1,'big'))

        im = cv2.imread(imagefilename,cv2.IMREAD_GRAYSCALE)
        imlist = im.reshape(-1).tolist()
        imarray = array('B', imlist)
        impackfile.write(imarray.tobytes())
    else:
        print(" FAILED TO PACK IMAGE ",i," :",imagefilename)


impackfile.close()
lblpackfile.close()
