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

text = convert_pdf_to_txt('AC010M00.02 EDs and Panel Layouts-R00.pdf')
inputpdf = PdfFileReader(open('AC010M00.02 EDs and Panel Layouts-R00.pdf','rb'))
first = re.split("next page:",text.lower())
page_numbers = []
for m in range(1,inputpdf.numPages):
    page_numbers.append(re.split("\n\n",re.split("page:",first[m])[1].strip())[0])
output = open("AC010M00.02 EDs and Panel Layouts-R00/AC010M00.02 EDs and Panel Layouts-R00.txt","w")
output.write(text)
output.close()
try:
    n = 1
    for page_number in page_numbers:
        filename =  "AC010M00.02 EDs and Panel Layouts-R00/images/PageNo "+str(page_number)+".jpg"
        os.rename('AC010M00.02 EDs and Panel Layouts-R00/images/PageNo '+str(n)+'.jpg',filename)
        n=n+1
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




