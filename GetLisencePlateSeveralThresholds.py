# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################

dir=""
dirname= dir +"test4Training\\images"
#dirname_labels = "testTest\\labels"
dirname_labels = dir +"test4Training\\labels"
#dirname_training=dir + "test4Training\\images"
dirname_thresolds=dir + "test4Training\\thresolds"

######################################################################

import pytesseract

import numpy as np

import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

X_resize=220
Y_resize=70

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
def ThresholdStable(image):
    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug 12 21:04:48 2022
    Author: Alfonso Blanco García
    
    Looks for the threshold whose variations keep the image STABLE
    (there are only small variations with the image of the previous 
     threshold).
    Similar to the method followed in cv2.MSER
    https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
    """
    
    import cv2
    import numpy as np
   

    thresholds=[]
    Repes=[]
    Difes=[]
    
    gray=image 
    grayAnt=gray

    ContRepe=0
    threshold=0
    for i in range (255):
        
        ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
        Dife1 = grayAnt - gray1
        Dife2=np.sum(Dife1)
        if Dife2 < 0: Dife2=Dife2*-1
        Difes.append(Dife2)
        if Dife2<22000: # Case only image of license plate
        #if Dife2<60000:    
            ContRepe=ContRepe+1
            
            threshold=i
            grayAnt=gray1
            continue
        if ContRepe > 0:
            
            thresholds.append(threshold) 
            Repes.append(ContRepe)  
        ContRepe=0
        grayAnt=gray1
    thresholdMax=0
    RepesMax=0    
    for i in range(len(thresholds)):
        #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
        if Repes[i] > RepesMax:
            RepesMax=Repes[i]
            thresholdMax=thresholds[i]
            
    #print(min(Difes))
    #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
    return thresholdMax

 
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
    
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

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
    Histograms=[]
    TFs=[]
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
        # Set total number of bins in the histogram

        bins_num = 256
            
        # Get the image histogram
            
        hist, bin_edges = np.histogram(image, bins=bins_num)
           
        #print(hist.shape)   
        Histograms.append(hist)
        TF =abs(np.fft.fft2(gray))
        #https://stackoverflow.com/questions/37152031/numpy-remove-a-dimension-from-np-array
        #TF=TF[:, :, 0]
        #print(TF.shape)
        TF=TF.flatten()
        TFs.append(TF)      
    return TFs, Histograms, imagesLicense, imagesLicenseFlat

def loadlabelsRoboflow (dirname ):
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
 ########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
     Thresholds=[]
     
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
         
         
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                
                 image = cv2.imread(filepath)
                 
                 #https://towardsdatascience.com/extract-text-from-memes-with-python-opencv-tesseract-ocr-63c2ccd72b69
                 gray= cv2.bilateralFilter(image,5, 55,60)
                
                 gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                 
                 SumBrightness=np.sum(image)
                 
                 Desv=np.std(image)
                 
                 if Desv < 45:
                     print("Image of " + License + " with low standard deviation, will be difficult to be recognized")
                 
                 
                 images.append(gray)
                 Licenses.append(License)
                 
                 
                
                 Cont+=1
     
     return images, Licenses
###########################################################
# MAIN
##########################################################

labels=loadlabelsRoboflow(dirname_labels)

imagesComplete, Licenses=loadimagesRoboflow(dirname)

TFs, Histograms, imagesLicense, imagesLicenseFlat= loadimagesOnlyLicense (imagesComplete, labels)

images=imagesLicense

print("Number of imagenes : " + str(len(images)))
print("Number of  labels : " + str(len(labels)))
print("Number of   licenses : " + str(len(Licenses)))

TotHits=0
TotFailures=0

NumberImageOrder=0

for i in range (len(images)):
        
        NumberImageOrder=NumberImageOrder+1
        
        
        gray=images[i]
        #cv2.imshow("Prueba", gray)
        #cv2.waitKey() 
      
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
        License=Licenses[i]
        SwEncontrado=0
         
        #
        # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
        #
             
        gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2) 
        text = pytesseract.image_to_string(gray1, lang='eng',  \
            config='--psm 13 --oem 3')
       
        text = ''.join(char for char in text if char.isalnum())
        
        if text==Licenses[i]:
                print(text + "  Hit with adaptive Threshold Mean and THRESH_BINARY" )
                TotHits=TotHits+1
                SwEncontrado=1
                
        if SwEncontrado==0:
             gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                 cv2.THRESH_BINARY_INV,11,2) 
             text = pytesseract.image_to_string(gray1, lang='eng',  \
                 config='--psm 13 --oem 3')
            
             text = ''.join(char for char in text if char.isalnum())
             
             if text==Licenses[i]:
                     print(text + "  Hit with adaptive Threshold  Mean and THRESH_BINARY_INV" )
                     TotHits=TotHits+1
                     SwEncontrado=1  
                     
        if SwEncontrado==0:
            
             gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv2.THRESH_BINARY,11,2)  
             text = pytesseract.image_to_string(gray1, lang='eng',  \
                 config='--psm 13 --oem 3')
            
             text = ''.join(char for char in text if char.isalnum())
             
             if text==Licenses[i]:
                     print(text + "  Hit with adaptive Threshold Gaussian and THRESH_BINARY" )
                     TotHits=TotHits+1
                     SwEncontrado=1  
                     
        if SwEncontrado==0:
            
             gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv2.THRESH_BINARY_INV,11,2)  
             text = pytesseract.image_to_string(gray1, lang='eng',  \
                 config='--psm 13 --oem 3')
            
             text = ''.join(char for char in text if char.isalnum())
             
             if text==Licenses[i]:
                     print(text + "  Hit with adaptive Threshold Gaussian and THRESH_BINARY_INV" )
                     TotHits=TotHits+1
                     SwEncontrado=1  
                    
        if SwEncontrado==0:
            
             #   Otsu's thresholding
             ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
             text = pytesseract.image_to_string(gray1, lang='eng',  \
                 config='--psm 13 --oem 3')
            
             text = ''.join(char for char in text if char.isalnum())
             
             if text==Licenses[i]:
                     print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_BINARY_INV" )
                     TotHits=TotHits+1
                     SwEncontrado=1   
                     
        if SwEncontrado==0:
            
            # Otsu's thresholding after Gaussian filtering
             blur = cv2.GaussianBlur(gray,(5,5),0)
             ret3,gray1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
             
             text = pytesseract.image_to_string(gray1, lang='eng',  \
                 config='--psm 13 --oem 3')
            
             text = ''.join(char for char in text if char.isalnum())
             
             if text==Licenses[i]:
                     print(text + "  Hit with Otsu's thresholding of cv2 , Gaussian filtering and THRESH_BINARY" )
                     TotHits=TotHits+1
                     SwEncontrado=1               
                     
        if SwEncontrado==0:
            
            threshold=OTSU_Threshold(gray)
            #https://omes-va.com/simple-thresholding/
            ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC)
            text = pytesseract.image_to_string(gray1, lang='eng',  \
                config='--psm 13 --oem 3')
           
            text = ''.join(char for char in text if char.isalnum())
            
            if text==Licenses[i]:
                    print(text + "  Hit with OTSU and THRESH_TRUNC" )
                    TotHits=TotHits+1
                    SwEncontrado=1
                
        if SwEncontrado==0:
             ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
             text = pytesseract.image_to_string(gray1, lang='eng',  \
                 config='--psm 13 --oem 3')
             text = ''.join(char for char in text if char.isalnum())
             
             if text==Licenses[i]:
                     print(text + "  Hit with OTSU and THRESH_TOZERO" )
                     TotHits=TotHits+1
                     SwEncontrado=1
                     
        if SwEncontrado==0:
             ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO_INV) 
             text = pytesseract.image_to_string(gray1, lang='eng',  \
                 config='--psm 13 --oem 3')
             text = ''.join(char for char in text if char.isalnum())
             
             if text==Licenses[i]:
                     print(text + "  Hit with OTSU and THRESH_TOZERO_INV" )
                     TotHits=TotHits+1
                     SwEncontrado=1
                     
        if SwEncontrado==0:
              ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY) 
              text = pytesseract.image_to_string(gray1, lang='eng',  \
                  config='--psm 13 --oem 3')
              text = ''.join(char for char in text if char.isalnum())
              
              if text==Licenses[i]:
                      print(text + "  Hit with OTSU and THRESH_BINARY" )
                      TotHits=TotHits+1
                      SwEncontrado=1
                      
        if SwEncontrado==0:
               ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY_INV) 
               text = pytesseract.image_to_string(gray1, lang='eng',  \
                   config='--psm 13 --oem 3')
               text = ''.join(char for char in text if char.isalnum())
               
               if text==Licenses[i]:
                       print(text + "  Hit with OTSU and THRESH_BINARY_INV" )
                       TotHits=TotHits+1
                       SwEncontrado=1
                       
        if SwEncontrado==0:
           
           ####################################################
           # experimental formula based on the brightness
           # of the whole image 
           ####################################################
           
           SumBrightness=np.sum(imagesComplete[i])  
           threshold=(SumBrightness/177600.00) 
           
           #####################################################
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit with  Brightness and THRESH_TRUNC" )
                   TotHits=TotHits+1
                   SwEncontrado=1
                   
        if SwEncontrado==0:
            
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit with Brightness and THRESH_BINARY" )
                   TotHits=TotHits+1
                   SwEncontrado=1
                   
        if SwEncontrado==0:
           
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY_INV) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit with Brightness and THRESH_BINARY_INV" )
                   TotHits=TotHits+1
                   SwEncontrado=1
                   
        if SwEncontrado==0:
           
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit with  Brightness and THRESH_TOZERO" )
                   TotHits=TotHits+1
                   SwEncontrado=1
                   
        if SwEncontrado==0:
           
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO_INV) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit with  Brightness and THRESH_TOZERO_INV" )
                   TotHits=TotHits+1
                   SwEncontrado=1
                   
        if SwEncontrado==0:
           threshold=ThresholdStable(gray)
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit con Stable y THRESH_TRUNC" )
                   TotHits=TotHits+1
                   SwEncontrado=1
                   
        if SwEncontrado==0:
           
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit con Stable y THRESH_BINARY" )
                   TotHits=TotHits+1
                   SwEncontrado=1
        
        if SwEncontrado==0:
           
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY_INV) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit with Stable y THRESH_BINARY_INV" )
                   TotHits=TotHits+1
                   SwEncontrado=1            
        
        if SwEncontrado==0:
           
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit with Stable y THRESH_TOZERO" )
                   TotHits=TotHits+1
                   SwEncontrado=1
        
        if SwEncontrado==0:
           
           ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO_INV) 
           text = pytesseract.image_to_string(gray1, lang='eng',  \
               config='--psm 13 --oem 3')
           text = ''.join(char for char in text if char.isalnum())
           
           if text==Licenses[i]:
                   print(text + "  Hit con Stable y THRESH_TOZERO_INV" )
                   TotHits=TotHits+1
                   SwEncontrado=1              
        
        if SwEncontrado==0:
          
           TotFailures=TotFailures+1
        
      
print("")           
print("Total Hits = " + str(TotHits ))
print("Total Failures = " + str(TotFailures ))
      
                 
        