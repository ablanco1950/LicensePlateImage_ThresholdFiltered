# LicensePlateImage_ThresholdFiltered
From some files of images and labels obtained by applying the project presented at https://github.com/ashok426/Vehicle-number-plate-recognition-YOLOv5, the images of license plates are filtered through a threshold that allows a better recognition of the license plate numbers by pytesseract.

Requirements:

The tests have been carried out on a computer with Windows11

Must installed the packages corresponding to :

import pytesseract

import numpy as np

import cv2

import re

import os


It is convenient to have Spyder installed for the execution of the program and Anaconda for the installation of the aforementioned packages from the cmd.exe option

Functioning:

Once the files have been downloaded, the test1 folder wihch contains the images and labels to test, is decompressed with its subfolders:

- images (which contains the car images, (downloaded from https://roboflow.com/) and the labels obtained by applying yolov5 to those images.
- labels: which contains the coordinates of the boxes demarcated by yolov5 for the car and license plate objects. ( have been obtained by applying https://github.com/ashok426/Vehicle-number-plate-recognition-YOLOv5)

The test1 folder should be in the same directory as the GetNumberLicencePlate.py program.

Run:

GetNumberLicencePlate.py


For inspection, it presents the images of each car and then those of its license plate, you have to close them for the program to continue advancing. Sometimes the image stays on the taskbar, you have to click on it and close it.

By console the following messages are obtained:

Reading images from test1\images\

Reading labels from test1\labels\

Number of images : 7

Number of labels : 7

Car 1 the license plate is not recognized

Car 2 the license plate is not recognized

Car 3 the license plate is recognized : 2122267

Car 4 the license plate is recognized : VI47 JAR

Car 5 the license plate is recognized : WM64 UMA

Car 6 the license plate is recognized : PL GO 321

Car 7 the license plate is recognized : M5 CEU

Comments:

Car1 has excessive light exposure on the license plate, so the license plate image appears burnt.

Car 2 has a slanted license plate so it is not recognized by pytesseract

Car 6's registration is recognized, but it has the error that it has assigned the L that appears, to an I that appears in the real registration.

The chosen filtering method is the threshold method, after seeing in the article
https://aicha-fatrah.medium.com/improve-the-quality-of-your-ocr-information-extraction-ebc93d905ac4 the radical improvement by applying this filtering to the OCR recognition of a French national identity document.

However, this threshold level cannot be fixed for each image, so one is implemented based on the total brightness of the image (sum of the values ​​of each pixel) to which an adjustment factor and bias are applied. Although the result is not applicable to all images due to their different qualities and circumstances.

For license plate recognition it is necessary to filter the various license plate number formats in which they are presented. In the formats of license plates from roboflow images, a great variety and anarchy of formats is detected.

As the program operates on label files created with yolov5, it is convenient to test by installing the application https://github.com/ashok426/Vehicle-number-plate-recognition-YOLOv5, however when doing it on a windows11 computer I have had I have to make some changes and I have made some simplifications that may be of interest to someone who works in that environment and which I attach to this document.

On 05/23/2022 a new version is introduced:
=========================================
GetNumberLicensePlateV1.py

Which is intended by varying the threshold level in a loop above and below the experimental formula of the thresold, that the correct enrollment is obtained for the most part.

For its execution, it needs that in the same directory of the program there is the Test2 directory that is attached and that contains the images and labels of the test.

It also requires the TrueLicenses.txt file in which the true license plates go, arranged in the same order as the images, to allow verification.

The results are:

HIT: the true license plate has been detected as the majority among all those provided by pytesseract

DETECTED: The true license plate has been detected by pytesseract, but it is not the majority.

NOT DETECTED: pytesseract has detected license plates, but none match the real one

NOT RECOGNIZED: pytesseract could not detect a license plate number.

The pyteseract results are filtered according to some registration formats, not all, which allows debugging.

This is a very slow brute force process

On 07/04/2022 a new ML version is introduced:
=========================================
From a set of images of cars with the labels of their license plates (test4Training folder), a Y_train is formed consisting of the set of thresholds at which each license plate is recognized by pytesseract, the X_train is made up of all the pixels of each image and the engine it is the sklearn SVM that allows you to make predictions of the threshord that would require a photo license plate to be recognized by pytesseract.

The training file is created by running:
CreateThresoldTraining.py

the Y_train values are in the files, one by each license, en test4Training\thresholds

Executing:

GetNumberLicencePlateV6SVM.py

The predictions of first ten  licenses plate of test4Training are getting

Observations:

Only images for which pytesseract can recognize the license plate number by varying the threshold are considered.

The error rate is not good since it operates with a training of 114 thresholds, which is small. When executing GetNumberLicencePlateV6SVM,py for the first 10 license plates, license plate 1032148 appears confused as 1932148, however adjusting the parameters at the beginning of the program:
imgStart=1
imgEnd=3
With which the test is reduced to a single record and the training is increased by 9 records, the licenset appears correctly predicted.

The name of each image, label and thresold has been changed. originally assigned by roboflow, so that they correspond to the license plate  number, allowing to detect failures and successes in the tests

References:

https://aicha-fatrah.medium.com/improve-the-quality-of-your-ocr-information-extraction-ebc93d905ac4

https://roboflow.com/

https://github.com/ashok426/Vehicle-number-plate-recognition-YOLOv5

https://gist.github.com/endolith/334196bac1cac45a4893#

https://towardsdatascience.com/the-practical-guide-for-object-detection-with-yolov5-algorithm-74c04aac4843

https://github.com/ultralytics/yolov5/issues/2293
