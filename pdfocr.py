#!/usr/bin/env python
# coding: utf-8

# In[63]:


# sudo apt install tesseract-ocr -y
import cv2 
#get_ipython().system('pip install pytesseract')
import pytesseract
from pytesseract import Output
import os
import regex


# In[64]:


# img = cv2.imread('output107.jpg')
# height, width, channels = img.shape


# In[65]:


# Adding custom options
# custom_config = r'--oem 3 --psm 6'
# d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
# string =  pytesseract.image_to_string(img, config=custom_config) 


# In[66]:


# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# filename = 'savedImage.jpg'
  
# # Using cv2.imwrite() method
# # Saving the image
# cv2.imwrite(filename, img)


# In[ ]:





# In[67]:


# texts = d["text"]


# In[68]:


# import regex
# re.split('\n',re.split('page: ', string.lower())[1])


# In[69]:


# page_number = str(texts[-1])


# In[70]:


# page_number


# In[71]:


pathRel = "./"
pathAbs = ""

contents = os.listdir(pathRel)

jpgdocs = []
output_folder_paths = []
for i in range(0,len(contents)):
        if contents[i].lower().endswith(('.jpg'))==True:
            jpgdocs.append(contents[i])
            output_folder_paths.append(pathRel+str(contents[i][:-4]))


# In[ ]:


for jpgdoc in jpgdocs:
    img = cv2.imread(jpgdoc)
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
    texts = d["text"]
    filename =  "PageNo"+str(texts[-1])+".jpg"
    os.rename(jpgdoc,filename)


# In[ ]:

#Another Method:
#for jpgdoc in jpgdocs:
#    img = cv2.imread(jpgdoc)
#    custom_config = r'--oem 3 --psm 6'
#    string =  pytesseract.image_to_string(img, config=custom_config)
#    try:
#        page_number = re.split('\n',re.split('page: ', string.lower())[2])[0]
#        filename =  "PageNo"+str(page_number)+".jpg"
#        os.rename(jpgdoc,filename)
#    except:
#        pass


# In[ ]:





# In[ ]:




