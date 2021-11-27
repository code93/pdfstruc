#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Python code to find the co-ordinates of
# the contours detected in an image.
import numpy as np
import cv2
  
# Reading image
font = cv2.FONT_HERSHEY_COMPLEX
img2 = cv2.imread('PageNo 20.jpg', cv2.IMREAD_COLOR)
  
# Reading same image in another 
# variable and converting to gray scale.
img = cv2.imread('PageNo 20.jpg', cv2.IMREAD_GRAYSCALE)
  
# Converting image to a binary image
# ( black and white only image).
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
  
# Detecting contours in image.
contours, _= cv2.findContours(threshold, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
  
# Going through every contours found in the image.
for cnt in contours :
  
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
  
    # draws boundary of contours.
    cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 
  
    # Used to flatted the array containing
    # the co-ordinates of the vertices.
    n = approx.ravel() 
    i = 0
  
    for j in n :
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]
  
            # String containing the co-ordinates.
            string = str(x) + " " + str(y) 
  
            if(i == 0):
                # text on topmost co-ordinate.
                cv2.putText(img2, "Arrow tip", (x, y),
                                font, 0.5, (255, 0, 0))
                print("Arrow Coordinates:"+string)
            else:
                # text on remaining co-ordinates.
                cv2.putText(img2, string, (x, y), 
                          font, 0.5, (0, 255, 0))
                print("other Coordinates"+string)
        i = i + 1

        
cv2.imwrite('imageCoordinates.jpg', img2)

# # Exiting the window if 'q' is pressed on the keyboard.
# if cv2.waitKey(0) & 0xFF == ord('q'): 
#     cv2.destroyAllWindows()


# In[2]:


# for getting the list of contours by Area:

# Sorting contours by Area

import cv2
import numpy as np
import pandas as pd

def get_contour_area(contours):
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

# Loading our image
image = cv2.imread("PageNo 20.jpg")
orginal_image = image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
contours, hierarchy =cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# print("contours Areas before sorting")
# print(get_contour_area(contours))

# sort contours Large to small
sorted_contours = sorted(contours, key=cv2.contourArea,reverse=True)

dataframe = pd.DataFrame()
# print("contours Areas after sorting")
# print(get_contour_area(sorted_contours))
area = get_contour_area(sorted_contours)
# print(len(area))
# print(len(sorted_contours))
#terate over our contours and draw one at a time
mylist = list(dict.fromkeys(area))



print(mylist)

# for knwing how many contours associated with an area

dataframe['area'] = area
dataframe['sorted_contours'] = sorted_contours
for name, group in dataframe.groupby('area'):
        print(name)
        print(len(group))
        print('\n')


# In[11]:


# Sorting contours by Area

import cv2
import numpy as np
import pandas as pd

def get_contour_area(contours):
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

# Loading our image
image = cv2.imread("PageNo 20.jpg")
orginal_image = image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
contours, hierarchy =cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# print("contours Areas before sorting")
# print(get_contour_area(contours))

# sort contours Large to small
sorted_contours = sorted(contours, key=cv2.contourArea,reverse=True)

dataframe = pd.DataFrame()
# print("contours Areas after sorting")
# print(get_contour_area(sorted_contours))
area = get_contour_area(sorted_contours)
# print(len(area))
# print(len(sorted_contours))
#terate over our contours and draw one at a time
mylist = list(dict.fromkeys(area))
# print(len(mylist))

# for i in mylist:
#     globals()[str(i)] = []
    
# print(len(mylist))
# for a in area:
#     for i in mylist:
#         if a == i:

# print(len(area),len(sorted_contours))

# print(dataframe)

# for j in mylist:
#     i = 0
#     for c in sorted_contours:
#         if (area[i]==j):
#             cv2.drawContours(orginal_image,[c],-1,(0,0,0),3)
#             cv2.imwrite("contours_Area_"+str(j)+".jpg",orginal_image)
#         i =  i+1
#         orginal_image = image

font = cv2.FONT_HERSHEY_COMPLEX

canvas = np.zeros((300, 300, 3), dtype="uint8")

l = 0
for c in sorted_contours:
    i = 0
    if (area[l]>300):
        cv2.drawContours(image,[c],-1,(0,0,0),3)
#         string = str(c[0][0][0])+','+str(c[0][0][1])+"    "
#         cv2.putText(orginal_image, string, (c[0][0][0],c[0][0][1]), 
#                              font, 0.5, (0, 255, 0))
#         string = str(c[1][0][0])+','+str(c[1][0][1])
#         cv2.putText(orginal_image, string, (c[1][0][0],c[1][0][1]), 
#                              font, 0.5, (0, 255, 0)) 
# #         n = approx.ravel()

        
#         for j in n :
#                 x = n[i]
#                 y = n[i + 1]

#                 # String containing the co-ordinates.
#                 string = str(x) + " " + str(y) 
#                 print(string)
               
#                 # text on remaining co-ordinates.
#                 cv2.putText(orginal_image, string, (x, y), 
#                             font, 0.5, (0, 255, 0)) 
               
        
#         i = i + 1
    l =  l+1
    
image[image != 0] = 255 # change everything to white where pixel is not black
kernel = np.ones((5,5), np.uint8)
kernel_sharpening = np.array(([-1,-1,-1],[-1,9,-1],[-1,-1,-1]))
sharpened = cv2.filter2D(image, -1, kernel_sharpening)
dilation = cv2.dilate(image, kernel, iterations = 1)
erosion = cv2.erode(dilation, kernel, iterations = 1)
cv2.imwrite("contours_Area.jpg",erosion)

# src1 = cv2.imread("contours_Area.jpg");
# src2 = cv2.imread("hocr-png.jpeg");

# dtype = -1;
# src3 = cv2.subtract(src1,src2);

# cv2.imwrite("Subtracted Image.jpg",src3)
# i = 0
#         for j in n :
#             if(i % 2 == 0):
#                 x = n[i]
#                 y = n[i + 1]

#                 # String containing the co-ordinates.
#                 string = str(x) + " " + str(y) 

#                 if(i == 0):
#                     # text on topmost co-ordinate.
#                     cv2.putText(orginal_image, "Arrow tip", (x, y),
#                                     font, 0.5, (255, 0, 0)) 
#                 else:
#                     # text on remaining co-ordinates.
#                     cv2.putText(orginal_image, string, (x, y), 
#                               font, 0.5, (0, 255, 0)) 
#             i = i + 1
    
# Showing the final image.
# cv2.imwrite('imageFinal.jpg', img2) 


# print(type(dataframe['area']), type(dataframe['sorted_contours']))
# grouped_df = dataframe.groupby("area")['sorted_contours']
# grouped_df = grouped_df.agg("sorted_contours": "nunique")
# grouped_df = grouped_df.reset_index()
# print(grouped_df)
# g = dataframe.groupby('area')['sorted_contours']

# dataframe['area'] = area
# dataframe['sorted_contours'] = sorted_contours
# for name, group in dataframe.groupby('area'):
#     if len(group)==36:
#         print(name)
#         print(len(group))
#         print('\n')


# In[32]:


#how-to-detect-diagram-region-and-extractcrop-it-from-a-research-papers-image

# Load image, grayscale, Otsu's threshold
image = cv2.imread('PageNo 20.jpg')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Dilate with horizontal kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,10))
dilate = cv2.dilate(thresh, kernel, iterations=2)

# Find contours and remove non-diagram contours
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if w/h > 2 and area > 10000:
        cv2.drawContours(dilate, [c], -1, (0,0,0), -1)

# Iterate through diagram contours and form single bounding box
boxes = []
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    boxes.append([x,y, x+w,y+h])

boxes = np.asarray(boxes)
x = np.min(boxes[:,0])
y = np.min(boxes[:,1])
w = np.max(boxes[:,2]) - x
h = np.max(boxes[:,3]) - y

# Extract ROI
cv2.rectangle(image, (x,y), (x + w,y + h), (36,255,12), 3)
ROI = original[y:y+h, x:x+w]

cv2.imwrite('image.jpg', image)
cv2.imwrite('thresh.jpg', thresh)
cv2.imwrite('dilate.jpg', dilate)
cv2.imwrite('ROI.jpg', ROI)
cv2.waitKey()


# In[6]:


# finding Contours ApproxPolyDP

import numpy as np
import cv2

image = cv2.imread('PageNo 20.jpg')
orig_image = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

contours, hierarchy =cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

#Iterate through each Contour and compute the boudning rectangle
# for c in contours:
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(orig_image(x,y), (x+w,y+h), (0,0,255), 2)
#     cv2.imwrite('Bounding Rectangle.jpg', orig_image)


# Iterate through each contour and compute the approx contour

for c in contours:
    accuracy = 0.03 * cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(image,[approx],0,(0,255,0),2)

cv2.imwrite('Approx Poly DP.jpg', image)


# In[9]:


# Convex Hull

import numpy as np
import cv2

image = cv2.imread('PageNo 20.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray,176,255,0)

contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

n = len(contours) - 1
contours = sorted(contours,key = cv2.contourArea,reverse=False)[:n]

# iterate through contours and draw the convex hull

for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image,[hull],0,(0,255,0),2)
    
cv2.imwrite('Convex Hull.jpg',image)


# In[14]:


import cv2
import numpy as np
def f(x): return

img = cv2.imread('PageNo 20.jpg')
cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cimg,50,150,apertureSize=3)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 200,param2=30)
cv2.imwrite('Circles.jpg',img)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    X = circles

print(X)


# In[28]:


import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
pipeline = keras_ocr.pipeline.Pipeline()
def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(img)

img = inpaint_text('PageNo 20.jpg',pipeline)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('MaskedImage.jpg',image)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




