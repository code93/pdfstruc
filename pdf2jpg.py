#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
# !pip install textract
# import textract
#get_ipython().system('pip install pdf2image')
from pdf2image import convert_from_path
#get_ipython().system('pip install PyPDF3')
from PyPDF3 import PdfFileWriter,PdfFileReader
import regex
import regex as re
import cv2
import numpy as np
from decouple import config

# In[39]:


pathRel = "./"
pathAbs = ""

contents = os.listdir(pathRel)

pdfdocs = []
output_folder_paths = []
for i in range(0,len(contents)):
        if contents[i].lower().endswith(('.pdf'))==True:
            pdfdocs.append(contents[i])
            output_folder_paths.append(pathRel+str(contents[i][:-4]))


# In[40]:


# pdf_file_paths = []
# for pdfdoc in pdfdocs:
#     absolute = pathRel+str(pdfdoc)
#     pdf_file_paths.append(absolute)
# #     text = textract.process(absolute)
# #     files.append(text)
# len(pdfdocs)


# In[41]:


for j in range(0,len(pdfdocs)):
    if os.path.isdir(output_folder_paths[j]) is False:
        os.mkdir(output_folder_paths[j])
        os.mkdir(str(output_folder_paths[j])+"/images")
        inputpdf = PdfFileReader(open(str(pdfdocs[j]),'rb'))
        maxPages = inputpdf.numPages
        i = 1
        for page in range(1,maxPages,10):
            pil_images = convert_from_path(str(pdfdocs[j]), dpi=300, first_page=page, 
                                           last_page=min(page+10-1,maxPages),fmt='jpg', 
                                           thread_count=1,userpw=None,use_cropbox=False, strict=False)
            for image in pil_images:
                image.save(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i)+'.jpg','JPEG')
                if config("use_opencv")=='true':
                    #how-to-detect-diagram-region-and-extractcrop-it-from-a-research-papers-image
                    try:
                        # Load image, grayscale, Otsu's threshold
                        if os.path.isdir(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i)) is False:
                            os.mkdir(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i))
                        image = cv2.imread(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i)+'.jpg')
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

                        cv2.imwrite(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i)+'/'+'image.jpg', image)
                        cv2.imwrite(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i)+'/'+'thresh.jpg', thresh)
                        cv2.imwrite(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i)+'/'+'dilate.jpg', dilate)
                        cv2.imwrite(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i)+'/'+'ROI.jpg', ROI)
                    except:
                        pass
                i = i + 1


from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from io import StringIO

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text
if config("use_regex_for_page_no")=='true':
    try:
        document = "AC010M00.02 EDs and Panel Layouts-R00.pdf"
        doc = document[:-4]
        text = convert_pdf_to_txt(document)
        inputpdf = PdfFileReader(open(document,'rb'))
        first = re.split("next page:",text.lower())
        output = open(str(doc)+"/"+str(doc)+".txt","w")
        output.write(text)
        output.close()
        page_numbers = []
        for m in range(1,inputpdf.numPages):
            page_numbers.append(re.split("\n\n",re.split("page:",first[m])[1].strip())[0])
        output = open(str(doc)+"/"+str(doc)+".txt","w")
        output.write(text)
        output.close()
        try:
            n = 1
            for page_number in page_numbers:
                filename =  str(doc)+"/images/PageNo "+str(page_number)+".jpg"
                os.rename(str(doc)+'/images/PageNo '+str(n)+'.jpg',filename)
                os.rename(str(doc)+'/images/PageNo '+str(n), filename[:-4])
                n=n+1
        except:
            pass
    except:
        pass
# In[23]:


# os.mkdir(pathRel+str(pdfdocs[0][:-4]))


# In[24]:


# pages = convert_from_path('AC010M00.02 EDs and Panel Layouts-R00.pdf')


# In[25]:


# inputpdf = PdfFileReader(open(str(pdfdocs[0]),'rb'))
# maxPages = inputpdf.numPages


# In[26]:


# print(maxPages)


# In[32]:


# i = 1
# for page in range(1,maxPages,10):
#     pil_images = convert_from_path(str(pdfdocs[0]), dpi=200, first_page=page, 
#                                    last_page=min(page+10-1,maxPages),fmt='jpg', 
#                                    thread_count=1,userpw=None,use_cropbox=False, strict=False)
#     for image in pil_images:
#         image.save(str(output_folder_paths[0])+'/'+'output'+str(i)+'.jpg','JPEG')
#         i = i + 1


# In[ ]:




