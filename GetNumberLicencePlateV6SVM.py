# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################
imgStart=0
imgEnd=11


#dirname = "testTest\\images"
dirname="test4Training\\images"
#dirname_labels = "testTest\\labels"
dirname_labels = "test4Training\\labels"
dirname_training="test4Training\\images"
dirname_thresolds="test4Training\\thresolds"

######################################################################

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
bias=4.3

import numpy as np

import cv2


import os
import re

import imutils

#####################################################################
"""
Copied from https://gist.github.com/endolith/334196bac1cac45a4893#

other source:
    https://stackoverflow.com/questions/46084476/radon-transformation-in-python
"""



from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft



try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):
    
   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
   
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency
#####################################################################

def loadThresolderTraining(dirname, imgStart, imgEnd):
    thresoldpath = dirname + "\\"
    
    
    arry=[""]
   
   
    print("Reading thresolds from ",thresoldpath)
    
    Conta=0
    ContFirst=0
    for root, dirnames, filenames in os.walk(thresoldpath):
        
               
        for filename in filenames:
            
            if re.search("\.(txt)$", filename):
                Conta=Conta+1
                #thresold only to training
                if Conta > imgStart and Conta < imgEnd:
                    continue
                filepath = os.path.join(root, filename)
              
              
                f=open(filepath,"r")
               
               
                for linea in f:
                    
                    lineadelTrain =linea.split(",")
                    
                    #if (float(lineadelTrain[2])) > 150.0:continue
                   
                   
                    if ContFirst==0:
                        ContFirst=1
                       
                        arry[0]=float(lineadelTrain[2])
                    else:
                        arry.append(float(lineadelTrain[2]))
                        
              
                f.close() 
               
                  
                
    
    
   
    Y_train=np.array(arry)
  
    return  Y_train



#########################################################################
def loadimages (dirname, imgStart, imgEnd, OptionTrainingTest ):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    imgpath = dirname + "\\"
    
    images = []
    imagesFlat=[]
    Licenses=[]
    arr=[]
    Conta=0
    ContFirst=0
    print("Reading imagenes from ",imgpath)
    NumImage=-2
    
    
    for root, dirnames, filenames in os.walk(imgpath):
        
        
        NumImage=NumImage+1
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                Conta=Conta+1
                # case test
                if OptionTrainingTest == 0:
                   if Conta > imgStart and Conta < imgEnd:
                       pp=0
                   else:
                       continue
                else:
                  if Conta > imgStart and Conta < imgEnd:
                      continue
                filepath = os.path.join(root, filename)
                License=filename[:len(filename)-4]
                image = cv2.imread(filepath)
               
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                 
                
                images.append(gray)
                #imagesFlat.append(gray.flatten())
                Licenses.append(License)
                         
    return images, Licenses
 #########################################################################
def loadlabels (dirname, imgStart, imgEnd, OptionTestTraining ):
 #########################################################################
 
 ########################################################################  
     lblpath = dirname + "\\"
     
     labels = []
    
     Conta=0
     print("Reading labels from ",lblpath)
     
     
     
     for root, dirnames, filenames in os.walk(lblpath):
         
                
         for filename in filenames:
             
             if re.search("\.(txt)$", filename):
                 Conta=Conta+1
                 # case test
                 if OptionTestTraining == 0:
                    if Conta > imgStart and Conta < imgEnd:
                        pp=0
                    else:
                        continue
                 else:
                   if Conta > imgStart and Conta < imgEnd:
                       continue
                 filepath = os.path.join(root, filename)
               
                 f=open(filepath,"r")

                 ContaLin=0
                 for linea in f:
                     
                     lineadelTrain =linea.split(" ")
                     if lineadelTrain[0] == "0":
                         ContaLin=ContaLin+1
                         labels.append(linea)
                         break
                 f.close() 
                 if ContaLin==0:
                     print("Rare labels without tag 0 on " + filename )
                   
                 
     
     return labels
#########################################################################
def loadimagesOnlyLicense (images, labels):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    NumberImageOrder=0
    imagesLicense=[]
    imagesLicenseFlat=[]
    ContFirst=0
    for i in range (len(images)):
        
        NumberImageOrder=NumberImageOrder+1
        
        lineaLabel =labels[i].split(" ")
        
        # Meaning of fields in files labels
        #https://github.com/ultralytics/yolov5/issues/2293
        #
        x_center=float(lineaLabel[1])
        y_center=float(lineaLabel[2])
        width=float(lineaLabel[3])
        heigh=float(lineaLabel[4])
        
        
        
        
        x_start= x_center - width*0.5
        x_end=x_center + width*0.5
        
        y_start= y_center - heigh*0.5
        y_end=y_center + heigh*0.5
        
        X_start=int(x_start*416)
        X_end=int(x_end*416)
        
        Y_start=int(y_start*416)
        Y_end=int(y_end*416)
        
        
        
        # Clipping the boxes in two positions helps
        # in license plate reading
        X_start=X_start + 3   
        Y_start=Y_start + 2
        
        
        #print ("X_start " + str(X_start))
        #print ("X_end " + str(X_end))
        #print ("Y_start " + str(Y_start))
        #print ("Y_end " + str(Y_end))
        
        image=images[i]
        
        #cv2.imshow("Test ", image)
        
        #cv2.waitKey()
        
             
        gray=image[Y_start:Y_end, X_start:X_end]
        
        X_resize=220
        Y_resize=70
        gray=cv2.resize(gray,None,fx=1.78,fy=1.78,interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
        
        rotation, spectrum, frquency =GetRotationImage(gray)
        rotation=90 - rotation
        
        #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
        #      " Desviacion : " + str(DesvLic))
        
        if rotation !=0 and rotation !=90:
            #print("SE ROTA LA IMAGEN " + str(rotation) + " GRADOS")
            gray=imutils.rotate(gray,angle=rotation)
       
       
        imagesLicense.append(gray)
        imagesLicenseFlat.append(gray.flatten())
       
              
    return imagesLicense, imagesLicenseFlat

###########################################################
# MAIN
##########################################################

  

Y_train = loadThresolderTraining(dirname_thresolds, imgStart, imgEnd)


OptionTestTraining=0
imagesTest,  LicensesTest=loadimages(dirname, imgStart, imgEnd, OptionTestTraining)
labelsTest=loadlabels(dirname_labels, imgStart, imgEnd, OptionTestTraining)
imagesLicenseTest, imagesLicenseTestFlat=loadimagesOnlyLicense (imagesTest, labelsTest )

OptionTestTraining=1
imagesTraining, LicensesTraining = loadimages(dirname_training, imgStart, imgEnd, OptionTestTraining)
labelsTraining=loadlabels(dirname_labels, imgStart, imgEnd, OptionTestTraining)
imagesLicenseTraining, imagesLicenseTrainingFlat=loadimagesOnlyLicense (imagesTraining, labelsTraining )

X_train=imagesLicenseTrainingFlat
X_test=imagesLicenseTestFlat


print("Number of imagenes to test : " + str(len(imagesTest)))
print("Number of  labels to test  : " + str(len(labelsTest)))
print("Number of  Licenses to test : " + str(len(LicensesTest)))
print("Number of  thresolds training : " + str(len(Y_train)))

from sklearn.svm import SVC
import pickle #to save the model

from sklearn.multiclass import OneVsRestClassifier


 #https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
model =  OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True, max_iter=1000)) #Creates model instance here
Y_train=Y_train.astype(int)
#X_train=X_train.astype(int)

model.fit(X_train, Y_train) #fits model with training data


pickle.dump(model, open("./model.pickle", 'wb')) #save model as a pickled file
 
predictions=model.predict(X_test)
#
TotHits=0
TotFailures=0

NumberImageOrder=0

for i in range (len( imagesLicenseTest)):
    
    # Blur the ROI of the detected licence plate
    # pesimos resultados
    #gray1 = cv2.GaussianBlur(imagesLicenseTest[i] ,    (35,35),0)
    
    
    #cv2.imshow("Prueba", gray1)
    #cv2.waitKey()
   
    
    NumberImageOrder=NumberImageOrder+1
    threshold=predictions[i] 
    print("thresold = "+ str(threshold))
    ret, gray1=cv2.threshold( imagesLicenseTest[i],threshold,255,  cv2.THRESH_BINARY)
    
    
    
    text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 13 --oem 3') 
    text = ''.join(char for char in text if char.isalnum())
    LicenseTest=LicensesTest[i]
    #https://stackoverflow.com/questions/67857988/removing-newline-n-from-tesseract-return-values
    #print(text)
    if (text[0:len(LicenseTest)]==LicenseTest):
       print ("HIT the license is detected as " + text[0:len(LicenseTest)])
       TotHits=TotHits+1
    else:                                             
        # se admite que pueda exisir al principio una posicion sin informacion 
        if text[1:len(LicenseTest)+1]==LicenseTest :
            print ("HIT the license is detected as " + text[1:len(LicenseTest)+1])
            TotHits=TotHits+1
        else:
              print ("Error is detected " + text + " insted the true license  " + LicenseTest)
              TotFailures=TotFailures +1 
print("")   
print(" Total Hits = " + str(TotHits)) 
print(" Total failures = " + str(TotFailures)) 
    
