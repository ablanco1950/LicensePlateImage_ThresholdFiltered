# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""

import pytesseract
import numpy as np
from PIL import Image
import cv2
######################################################################
# PARAMETERS
######################################################################
dirname = "test2\\images"
dirname_labels = "test2\\labels"
dirname_licenses="TrueLicenses.txt"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
bias=4.3


X_resize=220
Y_resize=70


Incthreshold=1.0

ContLoopMax=400
######################################################################

import os
import re


#########################################################################
def loadimages (dirname ):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    imgpath = dirname + "\\"
    
    images = []
   
    
    print("Reading imagenes from ",imgpath)
    NumImage=-2
    
    
    for root, dirnames, filenames in os.walk(imgpath):
        
        
        NumImage=NumImage+1
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                
                filepath = os.path.join(root, filename)
               
                image = cv2.imread(filepath)
                #https://stackoverflow.com/questions/51823228/get-orientation-pytesseract-python3
                #https://stackoverflow.com/questions/54047116/getting-an-error-when-using-the-image-to-osd-method-with-pytesseract
                #https://stackoverflow.com/users/5617608/esraa-abdelmaksoud
                #print(pytesseract.image_to_osd(Image.open(filepath), lang='eng', config='--psm 0 -c min_characters_to_try=5'))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                               
                images.append(gray)
                
    
    return images
 #########################################################################
def loadlabels (dirname ):
 #########################################################################
 
 ########################################################################  
     lblpath = dirname + "\\"
     
     labels = []
    
    
     print("Reading labels from ",lblpath)
     
     
     
     for root, dirnames, filenames in os.walk(lblpath):
         
                
         for filename in filenames:
             
             if re.search("\.(txt)$", filename):
                
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
 
def loadTrueLicenses (dirname ): 
 f=open(dirname)
 licenses=[]
 Conta=0;
 for linea in f:
     # quitar el cr
     linea1=linea[0:len(linea)-1]
     licenses.append(linea1)
     Conta=Conta+1
 f.close() 
 return licenses 

##########################################3
# Bubble sort in Python
#https://www.programiz.com/dsa/bubble-sort
def bubbleSort(array1,array2):
    
  # loop to access each array element
  for i in range(len(array1)):

    # loop to compare array elements
    for j in range(0, len(array1) - i - 1):

      # compare two adjacent elements
      # change > to < to sort in descending order
      if array1[j] < array1[j + 1]:

        # swapping elements if elements
        # are not in the intended order
        temp = array1[j]
        temp2= array2[j]
        array1[j] = array1[j+1]
        array2[j] = array2[j+1]
        array1[j+1] = temp
        array2[j+1] = temp2 
###########################################################
# MAIN
##########################################################

images=loadimages(dirname)

labels=loadlabels(dirname_labels)

TrueLicenses=loadTrueLicenses(dirname_licenses)

print("Number of imagenes : " + str(len(images)))
print("Number of  labels : " + str(len(labels)))
print("Number of  true licenses : " + str(len(TrueLicenses)))

TotHits=0
TotDetect=0
TotDetectBad=0
TotNoDetect=0


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
   
    #cv2.imshow("Test ", image)
    
    #cv2.waitKey()
    
    SwEnd=0
    
    SumBrightness=np.sum(image)
    Desv=np.std(image)
    if Desv < 45:
        print("Image with low standard deviation, will be difficult to be recognized")
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightness) +   
    #      " Desviacion : " + str(Desv))
    threshold=(SumBrightness/177529.84) + bias
    #print("SumBrightness = " + str(SumBrightness))   
    #print(" threshold " + str(threshold))
    gray=image[Y_start:Y_end, X_start:X_end]
    
    
    gray=cv2.resize(gray,None,fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
   
    SumBrightnessLic=np.sum(gray)
    DesvLic=np.std(gray)
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
    #      " Desviacion : " + str(DesvLic))
    
    Conta=0
    ContLoop=0
    SwEnd=0
    Llicenses=[]
   
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
        text = ''.join(char for char in text if char.isalnum())
        
        
        Confidence=""
        Case=0
        
        # Special case with an extra digit in the first position and
        # 7 positions plus case 1063HFG
        if len(text) > 7:
            text=text[1:8]
           
        
        if len(text) > 6:
            #
            # Case de licencia formed by NNNNNNN 7 numeric digits
            #
            if (text[0] >= "0"  and text[0] <= "9"
                   and  text[1] >= "0"  and text[1] <= "9"
                   and  text[2] >= "0"  and text[2] <= "9"
                   and  text[3] >= "0"  and text[3] <= "9"
                   and (( text[4] >= "0"  and text[4] <= "9") or text[4] == " ")
                   and (( text[5] >= "0"  and text[5] <= "9")  or text[5] == " ")
                   and  text[6] >= "0"  and text[6] <= "9"):
                
                Case=1
            else:
                # 
                # Case of license plate format AAANNAA
                #
                 if (((text[0] >= "A"  and text[0] <= "Z") )
                        and  text[1] >= "A"  and text[1] <= "Z"
                        and  text[2] >= "A"  and text[2] <= "Z"
                        and  ((text[3] >= "0"  and text[3] <= "9"))
                        and  text[4] >= "0"  and text[4] <= "9" 
                      
                        and  text[5] >= "A"  and text[5] <= "Z"
                        and  text[6] >= "A"  and text[6] <= "Z"):
                        
                  Case=2
                 else:
                     # 
                     # Case of license plate format AANNAAA
                     #
                      if (((text[0] >= "A"  and text[0] <= "Z") )
                             and  text[1] >= "A"  and text[1] <= "Z"
                             and  text[2] >= "0"  and text[2] <= "9"
                             and  ((text[3] >= "0"  and text[3] <= "9"))
                             and  text[4] >= "A"  and text[4] <= "Z" 
                           
                             and  text[5] >= "A"  and text[5] <= "Z"
                             and  text[6] >= "A"  and text[6] <= "Z"):
                             
                       Case=3
                      else:
                            # 
                            # Case of license plate format AAAANNN
                            #
                             if (((text[0] >= "A"  and text[0] <= "Z") )
                                    and  text[1] >= "A"  and text[1] <= "Z"
                                    and  text[2] >= "A"  and text[2] <= "Z"
                                    and  ((text[3] >= "A"  and text[3] <= "Z"))
                                    and  text[4] >= "0"  and text[4] <= "9" 
                                  
                                    and  text[5] >= "0"  and text[5] <= "9"
                                    and  text[6] >= "0"  and text[6] <= "9"):
                                    #and  text[7] >= "0"  and text[7] <= "Z"):
                               Case=4
                             else:
                                    # 
                                    # Case of license plate format AAANAAA
                                    #
                                     if (((text[0] >= "A"  and text[0] <= "Z") )
                                            and  text[1] >= "A"  and text[1] <= "Z"
                                            and  text[2] >= "A"  and text[2] <= "Z"
                                            and  ((text[3] >= "0"  and text[3] <= "9"))
                                            and  text[4] >= "A"  and text[4] <= "Z" 
                                          
                                            and  text[5] >= "A"  and text[5] <= "Z"
                                            and  text[6] >= "A"  and text[6] <= "Z"):
                                            
                                      Case=5
                                     else:
                                             # 
                                             # Case of license plate format AANNAAA
                                             #
                                              if (((text[0] >= "A"  and text[0] <= "Z") )
                                                     and  text[1] >= "A"  and text[1] <= "Z"
                                                     and  text[2] >= "0"  and text[2] <= "9"
                                                     and  ((text[3] >= "0"  and text[3] <= "9"))
                                                     and  text[4] >= "A"  and text[4] <= "Z" 
                                                   
                                                     and  text[5] >= "A"  and text[5] <= "Z"
                                                     and  text[6] >= "A"  and text[6] <= "Z"):
                                                     
                                                  Case=6
                                              else:
                                                       # 
                                                       # Case of license plate format AANNAAA
                                                       #
                                                        if (((text[0] >= "0"  and text[0] <= "9") )
                                                               and  text[1] >= "0"  and text[1] <= "9"
                                                               and  text[2] >= "0"  and text[2] <= "9"
                                                               and  ((text[3] >= "0"  and text[3] <= "9"))
                                                               and  text[4] >= "A"  and text[4] <= "Z" 
                                                             
                                                               and  text[5] >= "A"  and text[5] <= "Z"
                                                               and  text[6] >= "A"  and text[6] <= "Z"):
                                                               
                                                             Case=7
                                                        else:
                                                                  # 
                                                                  # Case of license plate format AANNNN pero que se ha colado
                                                                  # un caracter en la primera posicion
                                                                   if (((text[1] >= "A"  and text[1] <= "Z") )
                                                                          and  text[2] >= "A"  and text[2] <= "Z"
                                                                          and  text[3] >= "0"  and text[3] <= "9"
                                                                          and  ((text[4] >= "0"  and text[4] <= "9"))
                                                                          and  text[5] >= "0"  and text[5] <= "9" 
                                                                        
                                                                          and  text[6] >= "0"  and text[6] <= "9"):
                                                                          
                                                                    Case=8
        # 
        # Case of license plate format AAANNN
        #
        else:
            if len(text) > 5:
              if (((text[0] >= "A"  and text[0] <= "Z") )
                     and  text[1] >= "A"  and text[1] <= "Z"
                     and  text[2] >= "A"  and text[2] <= "Z"
                     and  text[3] >= "0"  and text[3] <= "9"
                     and (( text[4] >= "0"  and text[4] <= "9") )
                    
                     and  text[5] >= "0"  and text[5] <= "9"):
                    
                 
                 Case=9
              else: 
                 # Format AANNNN
                 if (((text[0] >= "A"  and text[0] <= "Z") )
                        and  text[1] >= "A"  and text[1] <= "Z"
                        and  text[2] >= "0"  and text[2] <= "9"
                        and  text[3] >= "0"  and text[3] <= "9"
                        and (( text[4] >= "0"  and text[4] <= "9") )
                       
                        and  text[5] >= "0"  and text[5] <= "9"):
                       
                    
                    Case=10
            else:
                # Format ANAAA
                if len(text) > 4:
                  if (((text[0] >= "A"  and text[0] <= "Z") )
                         #pytesseract confuses 5 an S
                         and  ((text[1] >= "0"  and text[1] <= "9") or text[1]=="S")
                         and  text[2] >= "A"  and text[2] <= "Z"
                         and  text[3] >= "A"  and text[3] <= "Z"
                         and  text[4] >= "A"  and text[4] <= "Z"):
                                             
                     
                     Case=11
     
       
        
        TextConfidence=Confidence + " threshold : " + str(threshold)
        if Case > 0: 
            if Case==1 or Case==2 or Case==3 or Case==4 or Case==5 or Case==7:
                print ("Car " + str(NumberImageOrder) +  " the license plate is recognized as : " + text[0:7] + TextConfidence)
                Llicenses.append(text[0:7] )
                SwEnd=1
            else:
                if Case==8:
                    print ("Car " + str(NumberImageOrder) +  " the license plate is recognized as : " + text[1:7]+TextConfidence)
                    Llicenses.append(text[1:7] )
                    SwEnd=1
                else:
                    if Case==9 or Case==10:
                        print ("Car " + str(NumberImageOrder) +  " the license plate is recognized  as : " + text[0:6]+TextConfidence)
                        Llicenses.append(text[0:6] )
                        SwEnd=1
                    else:
                        if Case == 11 :
                           print ("Car " + str(NumberImageOrder) +  " the license plate is recognized as : " + text[0:5]+TextConfidence)
                           Llicenses.append(text[0:5] )
                        
       
            SwEnd=0
        
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
          
    LTotLicenses=[]
    LSumLicenses=[]
    TotLicences=0
    for y in range(len(Llicenses)):
        
        SwFounded=0
        for z in range(len(LTotLicenses)):
            if Llicenses[y]==LTotLicenses[z]:
                LSumLicenses[z]=LSumLicenses[z]+1
                TotLicences=TotLicences+1
                SwFounded=1
                break
        if SwFounded==0:
            LTotLicenses.append(Llicenses[y])
            LSumLicenses.append(1)
            TotLicences=TotLicences+1
            
    print("Licenses Car " + str(NumberImageOrder))  
    
    bubbleSort(LSumLicenses,LTotLicenses)
    TextDetect=""
    for w in range(len(LTotLicenses)):
        percent=LSumLicenses[w]/TotLicences
        #https://docs.python.org/3/tutorial/inputoutput.html
        print (LTotLicenses[w] + " {:2.2%}".format(percent))
        
        
        if LTotLicenses[w] == TrueLicenses[i]:
           
            if w==0:
                TotHits=TotHits+1
                TextDetect=" HIT" + " {:2.2%}".format(percent)
                #break
            else:
                TotDetect=TotDetect+1
                TextDetect=" DETECTED " + " {:2.2%}".format(percent)
                #break
    if TextDetect=="":
       if len(LTotLicenses)==0:
           TotNoDetect=TotNoDetect+1
           TextDetect= " DID NOT DETECT ANY LICENSE PLATE"
       else:
           TotDetectBad=TotDetectBad+1
           TextDetect= " ERROR, DETECTED OTHER LICENSE PLATE"
    print( "Car " + str(NumberImageOrder) +  " the  true license plate is: " + TrueLicenses[i] + TextDetect)    

print("")
print("")
print(" total HITS : " + str(TotHits))
print(" total DETECTED : " + str(TotDetect)) 
print(" total ERRONEOUS DETECTED : " + str(TotDetectBad)) 
print(" total DID NOT RECOGNIZED : " + str(TotNoDetect))     
        