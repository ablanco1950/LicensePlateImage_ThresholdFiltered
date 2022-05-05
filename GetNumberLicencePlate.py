# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""

import pytesseract
import numpy as np

import cv2
######################################################################
# PARAMETERS
######################################################################
dirname = "test1\\images"
dirname_labels = "test1\\labels"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
bias=4.3
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
###########################################################
# MAIN
##########################################################

images=loadimages(dirname)

labels=loadlabels(dirname_labels)

print("Number of imagenes : " + str(len(images)))
print("Number of  labels : " + str(len(labels)))

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
    X_start=X_start + 2   
    Y_start=Y_start + 2
    
    
    #print ("X_start " + str(X_start))
    #print ("X_end " + str(X_end))
    #print ("Y_start " + str(Y_start))
    #print ("Y_end " + str(Y_end))
    
    image=images[i]
   
    cv2.imshow("Test ", image)
    
    cv2.waitKey()
    
    SwEnd=0
    
    SumBrightness=np.sum(image)
    
    threshold=(SumBrightness/177529.84) + bias
    #print("SumBrightness = " + str(SumBrightness))   
    #print(" threshold " + str(threshold))
    gray=image[Y_start:Y_end, X_start:X_end]
    
   
    gray=cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    Conta=0
    while (SwEnd==0):
        
        #https://java2blog.com/cv2-threshold-python/
        #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
        
        #https://aicha-fatrah.medium.com/improve-the-quality-of-your-ocr-information-extraction-ebc93d905ac4
        ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY)
        
        cv2.imshow("Prueba", gray1)
        cv2.waitKey()
       
        
        text = pytesseract.image_to_string(gray1, lang='eng',  \
            config='--psm 13 --oem 3') 
        #https://stackoverflow.com/questions/67857988/removing-newline-n-from-tesseract-return-values
        text = text.replace("\n", " ")
        #https://www.delftstack.com/es/howto/python/python-remove-quotes-from-string/
        text = text.replace('"','')
        text = text.replace('~','')
        text = text.replace('|','')
        text = text.replace(']','')
        
        #print(text)
        #print(len(text))
        
        Case=0
        
        if len(text) > 6:
            #
            # Case de licencia formada por 7 digitos numericos
            #
            if (text[0] >= "0"  and text[0] <= "9"
                   and  text[1] >= "0"  and text[1] <= "9"
                   and  text[2] >= "0"  and text[2] <= "9"
                   and  text[3] >= "0"  and text[3] <= "9"
                   and (( text[4] >= "0"  and text[4] <= "9") or text[4] == " ")
                   and (( text[5] >= "0"  and text[5] <= "9")  or text[5] == " ")
                   and  text[6] >= "0"  and text[6] <= "9"):
                #text1=[]
                #for i in range(7):
                #    text1.append(text[i])
                
                #text =''.join(text1)
                Case=2
            else:
                # 
                # Case of license plate format AAA AAA
                #
                 if (((text[0] >= "0"  and text[0] <= "z") or text[0] == " ")
                        and  text[1] >= "0"  and text[1] <= "z"
                        and  text[2] >= "0"  and text[2] <= "z"
                        and  ((text[3] >= "0"  and text[3] <= "z") or text[3] == " ")
                        and  text[4] >= "0"  and text[4] <= "z" 
                      
                        and  text[5] >= "0"  and text[5] <= "z"
                        and  text[6] >= "0"  and text[6] <= "z"):
                        #and  text[7] >= "0"  and text[7] <= "z"):
                  Case=1
        
        # 
        # Case of license plate format AAAA AAA
        #
        if len(text) > 7:
          if (((text[0] >= "0"  and text[0] <= "z") or text[0] == " ")
                 and  text[1] >= "0"  and text[1] <= "z"
                 and  text[2] >= "0"  and text[2] <= "z"
                 and  text[3] >= "0"  and text[3] <= "z"
                 and (( text[4] >= "0"  and text[4] <= "z") or text[4] == " ")
                
                 and  text[5] >= "0"  and text[5] <= "z"
                 and  text[6] >= "0"  and text[6] <= "z"
                 and  text[7] >= "0"  and text[7] <= "z"):
             
           Case=3
     
        #
        # Case of license plate format AA AA NNN
        #
        if len(text) > 8:
            
            if (text[0] >= "0"  and text[0] <= "z"
                    and  text[1] >= "0"  and text[1] <= "z"
                    and  text[2] >= " "  
                    and  text[3] >= "0"  and text[3] <= "z"
                    and  text[4] >= "0"  and text[4] <= "z"
                    and  text[5] == " "       
                    and  text[6] >= "0"  and text[6] <= "z"
                    and  text[7] >= "0"  and text[7] <= "z"
                    and  text[8] >= "0"  and text[8] <= "z"):
                 
                 Case=4
         
        if Case > 0: 
            if Case==1:
                print ("Car " + str(NumberImageOrder) +  " the license plate is recognized : " + text[0:7])
                SwEnd=1
            else:
                if Case==2:
                    print ("Car " + str(NumberImageOrder) +  " the license plate is recognized : " + text[0:7])
                    SwEnd=1
                else:
                    if Case == 3 :
                       print ("Car " + str(NumberImageOrder) +  " the license plate is recognized : " + text[0:8])
                    else:
                        print ("Car " + str(NumberImageOrder) +  " the license plate is recognized : " + text[0:9])
                    SwEnd=1
        else:
            print("Car " + str(NumberImageOrder) +  " the license plate is not recognized  ")
            SwEnd=1
          
