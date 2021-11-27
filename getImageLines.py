
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

