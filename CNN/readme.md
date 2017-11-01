to train
========
python main.py --train


to test
=======
python main.py --test


to both train and test
======================
python main.py --train --test


to predict
==========
python main.py --predict --image-path pic.jpg


************************************************************************
ImageData class holds the data for the raw image given for prediction
it first calculates the rects (face(s) positions)
it also pre-processes the rects(faces)   gray ==> align ==> crop ==> intensity normalize
it also has showResults() which takes in predictions as argument, and shows the faces and respective expressions on the input image

deepnn.py holds the deep neural network core logic.
(conv => relu => maxpool ==> conv => relu => maxpool => fully connected ==> fully connected )

cohnKanadeDataset class houses training and testing datasets. It uses methods to convert a binary file to image dataset.
Dataset class is the type of ck.train and ck.test. It has methods to render slices of dataset in batches. Also shuffles if needed (by default).
