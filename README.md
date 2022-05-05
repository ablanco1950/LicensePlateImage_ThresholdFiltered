# LicensePlateImage_ThresholdFiltered
From some files of images and labels obtained by applying the project presented at https://github.com/ashok426/Vehicle-number-plate-recognition-YOLOv5, the images of license plates are filtered through a threshold that allows a better recognition of the license plate numbers by pytesseract.

Requirements:

The tests have been carried out on a computer with Windows11

Must installed the packages corresponding to :

import pytesseract

import numpy as np

import cv2

matter you

import re

import os


It is convenient to have Spyder installed for the execution of the program and Anaconda for the installation of the aforementioned packages from the cmd.exe option

Functioning:

Once the files have been downloaded, the test1 folder is decompressed with its subfolders:

- images (which contains the car images, (downloaded from https://roboflow.com/) and the labels obtained by applying yolov5 to those images.
- labels: which contains the coordinates of the boxes demarcated by yolov5 for the car and license plate objects. (supposed to have been obtained by applying https://github.com/ashok426/Vehicle-number-plate-recognition-YOLOv5)

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

For license plate recognition it is necessary to filter the various license plate number formats in which they are presented. In the formats of roboflow images, a great variety and anarchy of formats is detected.

As the program operates on label files created with yolov5, it is convenient to test by installing the application https://github.com/ashok426/Vehicle-number-plate-recognition-YOLOv5, however when doing it on a windows11 computer I have had I have to make some changes and I have made some simplifications that may be of interest to someone who works in that environment and which I attach to the document:

References:
https://aicha-fatrah.medium.com/improve-the-quality-of-your-ocr-information-extraction-ebc93d905ac4
https://roboflow.com/
https://github.com/ashok426/Vehicle-number-plate-recognition-YOLOv5)
https://github.com/ultralytics/yolov5/issues/2293
