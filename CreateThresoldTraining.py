# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################
dirname = "test4Training\\images"
dirname_labels = "test4Training\\labels"
dirname_thresolds="test4Training\\thresolds"
######################################################################

import pytesseract

import numpy as np
from PIL import Image
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
bias=4.3


X_resize=220
Y_resize=70

Incthreshold=1.0

ContLoopMax=400
######################################################################

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
#########################################################################
def loadimages (dirname ):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    imgpath = dirname + "\\"
    
    images = []
    Licenses=[]
   
    
    print("Reading imagenes from ",imgpath)
    NumImage=-2
    
    Cont=0
    for root, dirnames, filenames in os.walk(imgpath):
        
        
        NumImage=NumImage+1
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                Cont=Cont+1
                
                filepath = os.path.join(root, filename)
                License=filename[:len(filename)-4]
               
                image = cv2.imread(filepath)
               
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                               
                images.append(gray)
                Licenses.append(License)
    
    return images, Licenses
 #########################################################################
def loadlabels (dirname ):
 #########################################################################
 
 ########################################################################  
     lblpath = dirname + "\\"
     
     labels = []
    
    
     print("Reading labels from ",lblpath)
     
     
     Cont=0
     for root, dirnames, filenames in os.walk(lblpath):
         
                
         for filename in filenames:
             
             if re.search("\.(txt)$", filename):
                 Cont=Cont+1
                 #if Cont > 3: break
                 filepath = os.path.join(root, filename)
                 
                 
                 filepath = os.path.join(root, filename)
               
                 f=open(filepath,"r")

                 Conta=0
                 for linea in f:
                     
                     lineadelTrain =linea.split(" ")
                     if lineadelTrain[0] == "0":
                         Conta=Conta+1
                         labels.append(linea)
                         break
                 f.close() 
                 if Conta==0:
                     print("Rare labels without tag 0 on " + filename )
                   
                 
     
     return labels
# Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
def OTSU_Threshold(image):
# Set total number of bins in the histogram

    bins_num = 256
    
    # Get the image histogram
    
    hist, bin_edges = np.histogram(image, bins=bins_num)
   
    # Get normalized histogram if it is required
    
    #if is_normalized:
    
    hist = np.divide(hist.ravel(), hist.max())
    
     
    
    # Calculate centers of bins
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    
    weight1 = np.cumsum(hist)
    
    weight2 = np.cumsum(hist[::-1])[::-1]
   
    # Get the class means mu0(t)
    
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold
###########################################################
# MAIN
##########################################################




images, Licenses =loadimages(dirname)

labels=loadlabels(dirname_labels)


print("Number of imagenes : " + str(len(images)))
print("Number of  labels : " + str(len(labels)))
print("Number of   licenses : " + str(len(Licenses)))

TotHits=0
TotFailures=0


NumberImageOrder=0

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
        
        
        
        image=images[i]
        License=Licenses[i]
       
        #cv2.imshow("Test ", image)
        
        #cv2.waitKey()
        
        SwEnd=0
        
        lineaw=[]
        #lineaw.append(TrueLicenses[i])
        SumBrightness=np.sum(image)
        lineaw.append(str(SumBrightness))
        Desv=np.std(image)
        lineaw.append(str(Desv))
        if Desv < 45:
            print("Image with low standard deviation, will be difficult to be recognized")
        #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightness) +   
        #      " Desviacion : " + str(Desv))
        threshold=(SumBrightness/177529.84) + bias
        print("SumBrightness = " + str(SumBrightness) + " Desviacion = " + str(Desv))   
        #print(" threshold " + str(threshold))
        gray=image[Y_start:Y_end, X_start:X_end]
        
        
        gray=cv2.resize(gray,None,fx=1.78,fy=1.78,interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
       
        SumBrightnessLic=np.sum(gray)
        DesvLic=np.std(gray)
        rotation, spectrum, frquency =GetRotationImage(gray)
        rotation=90 - rotation
        #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
        #      " Desviacion : " + str(DesvLic))
        if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
          
            gray=imutils.rotate(gray,angle=rotation)
        #else:
        #    continue
        Conta=0
        ContLoop=0
        SwEnd=0
        Llicenses=[]
        SwEncontrado=0
        while (ContLoop < ContLoopMax):
      
            if ContLoop >ContLoopMax: break
            ContLoop=ContLoop+1
           
            #https://java2blog.com/cv2-threshold-python/
            #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
            
            #https://aicha-fatrah.medium.com/improve-the-quality-of-your-ocr-information-extraction-ebc93d905ac4
            
            
            ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY)
           
            #cv2.imshow("Prueba", gray1)
            #cv2.waitKey() 
           
            
            text = pytesseract.image_to_string(gray1, lang='eng',  \
                config='--psm 13 --oem 3')
            #new_gray1 = Image.fromarray(gray1)
            #text = tesserocr.image_to_text(new_gray1)
            text = ''.join(char for char in text if char.isalnum())
            print(text)
            if (text[0:len(License)]==License) or \
                (text[1:len(License)+1]==License)    :
               with open( dirname_thresolds+"\\" + License +".txt","w") as  w:
                    print("FOUNDED License Plate, thresold = " +str(threshold))
                    SwEncontrado=1
                    lineaw.append(str(threshold))
                    lineaWrite =','.join(lineaw)
                    lineaWrite=lineaWrite + "\n"
                    w.write(lineaWrite)
               w.close
               TotHits=TotHits+1
               break
            #
            # Halfway through the loop, 
            # it searches in the negative direction for the threshold
            #
            if ContLoop==ContLoopMax/2:
                threshold=threshold- Incthreshold*ContLoop
            if ContLoop>=ContLoopMax/2:
                threshold=threshold-Incthreshold
            else:
                threshold=threshold+Incthreshold
    
        #
        # if it is not found try to try with the OTSU thresholdr
        # 

        if SwEncontrado==0:
            threshold=OTSU_Threshold(image)
            ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY)
           
            #cv2.imshow("Prueba", gray1)
            #cv2.waitKey() 
           
            
            text = pytesseract.image_to_string(gray1, lang='eng',  \
                config='--psm 13 --oem 3 --dpi 300') 
           
            
            text = ''.join(char for char in text if char.isalnum())
            print(text)
            if (text[0:len(License)]==License) or \
                (text[1:len(License)+1]==License)    :
               with open( dirname_thresolds+"\\" + License +".txt","w") as  w:
                    print("FOUNDED by OTSU thresold = " +str(threshold))
                    SwEncontrado=1
                    lineaw.append(str(threshold))
                    lineaWrite =','.join(lineaw)
                    lineaWrite=lineaWrite + "\n"
                    w.write(lineaWrite)
                   
               w.close
               TotHits=TotHits+1
            else:
                TotFailures=TotFailures+1
print("")   
print(" Total Hits = " + str(TotHits)) 
print(" Total failures = " + str(TotFailures))            
        