#!/usr/bin/env python
# coding: utf-8

# In[37]:


import os
import shutil
# !pip install textract
# import textract
#!pip install pdf2image
from pdf2image import convert_from_path
#!pip install PyPDF3
from PyPDF3 import PdfFileWriter,PdfFileReader
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import regex as re
nlp = spacy.load("en_core_web_sm")


# In[28]:


pathRel = "./"
pathAbs = ""

contents = os.listdir(pathRel)

pdfdocs = []
output_folder_paths = []
for i in range(0,len(contents)):
        if contents[i].lower().endswith(('.pdf'))==True:
            pdfdocs.append(contents[i])
            output_folder_paths.append(pathRel+str(contents[i][:-4]))


# In[29]:


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


# In[30]:


# pdf_file_paths = []
# for pdfdoc in pdfdocs:
#     absolute = pathRel+str(pdfdoc)
#     pdf_file_paths.append(absolute)
# #     text = textract.process(absolute)
# #     files.append(text)
# len(pdfdocs)


# In[31]:


articles = []
for j in range(0,len(pdfdocs)):
    if os.path.isdir(output_folder_paths[j]) is False:
        os.mkdir(output_folder_paths[j])
        os.mkdir(str(output_folder_paths[j])+"/images")
        inputpdf = PdfFileReader(open(str(pdfdocs[j]),'rb'))
        try:
            doc = []
#             for i in range (0, inputpdf.numPages):
#                 doc.append(inputpdf.getPage(i).extractText())
#             def listToString(s): 
#                 str1 = "" 
#                 for ele in s: 
#                     str1 += ele
#                 return str1
#             text = listToString(doc)
            text = convert_pdf_to_txt(str(pdfdocs[j]))
            articles.append(text)
            doc = nlp(text)
            tokens = [token.text for token in doc]
            punctuation = punctuation + '\n'

            # text cleaning

            word_freq = {}

            stop_words = list(STOP_WORDS)

            for word in doc:
                if word.text.lower() not in stop_words:
                    if word.text.lower() not in punctuation:
                        if word.text not in word_freq.keys():
                            word_freq[word.text] = 1
                        else:
                            word_freq[word.text] += 1

            max_freq = max(word_freq.values())
            for word in word_freq.keys():
                word_freq[word]= word_freq[word] / max_freq

            sent_tokens = [sent for sent in doc.sents]
            sent_score = {}
            for sent in sent_tokens:
                for word in sent:
                    if word.text.lower() in word_freq.keys():
                        if sent not in sent_score.keys():
                            sent_score[sent] = word_freq[word.text.lower()]
                        else:
                            sent_score[sent] += word_freq[word.text.lower()]
                    # select 30% sentences with maximum score
            summary = nlargest(n = round(len(sent_score)*0.30), iterable=sent_score, key=sent_score.get)
            final_summary =  [word.text for word in summary]
            summary = " ".join(final_summary)
            # dealing with pdf file
            output = open(str(output_folder_paths[j])+'/'+"summary.txt","w")
            output.write(summary)
            output.close()
        except:
            pass
        finally:    
            maxPages = inputpdf.numPages

            i = 1
            for page in range(1,maxPages,10):
                pil_images = convert_from_path(str(pdfdocs[j]), dpi=200, first_page=page, 
                                               last_page=min(page+10-1,maxPages),fmt='jpg', 
                                               thread_count=1,userpw=None,use_cropbox=False, strict=False)
                for image in pil_images:
                    image.save(str(output_folder_paths[j])+'/images/'+'PageNo '+str(i)+'.jpg','JPEG')
                    i = i + 1


# In[32]:


#         inputpdf = PdfFileReader(open(str(pdfdocs[j]),'rb'))
           
#             for i in range (0, inputpdf.numPages):
#                 doc.append(inputpdf.getPage(i).extractText())
#             def listToString(s): 
#                 str1 = "" 
#                 for ele in s: 
#                     str1 += ele
#                 return str1
#             text = listToString(doc)


# In[33]:


#npr['Topic']= 8


# npr = pd.DataFrame()
# npr['Articles'] = articles
# npr['Topic'] = topic_results.argmax(axis=1)

if os.path.isdir("Topic_1") is False:
    os.mkdir("Topic_1")
if os.path.isdir("Topic_2") is False:
    os.mkdir("Topic_2")
if os.path.isdir("Topic_3") is False:
    os.mkdir("Topic_3")
if os.path.isdir("Topic_4") is False:
    os.mkdir("Topic_4")
if os.path.isdir("Topic_5") is False:
    os.mkdir("Topic_5")
if os.path.isdir("Topic_6") is False:
    os.mkdir("Topic_6")
if os.path.isdir("Topic_7") is False:
    os.mkdir("Topic_7")
if os.path.isdir("SSRN") is False:
    os.mkdir("SSRN")
# os.mkdir(pathRel+str(pdfdocs[0][:-4]))


# In[34]:


try:
    cv = CountVectorizer(max_df=0.9,min_df=2,stop_words='english')
    dtm = cv.fit_transform(articles)

    LDA = LatentDirichletAllocation(n_components=7,random_state=42)
    LDA.fit(dtm)
    topic_results = LDA.transform(dtm)
    for k in range(0,len(pdfdocs)):
        if topic_results[k].argmax()==0:
            shutil.copy(pdfdocs[k], "Topic_1")
        if topic_results[k].argmax()==1:
            shutil.copy(pdfdocs[k], "Topic_2")
        if topic_results[k].argmax()==2:
            shutil.copy(pdfdocs[k], "Topic_3")
        if topic_results[k].argmax()==3:
            shutil.copy(pdfdocs[k], "Topic_4")
        if topic_results[k].argmax()==4:
            shutil.copy(pdfdocs[k], "Topic_5")
        if topic_results[k].argmax()==5:
            shutil.copy(pdfdocs[k], "Topic_6")
        if topic_results[k].argmax()==6:
            shutil.copy(pdfdocs[k], "Topic_7")
except:
    pass

#shutil.copy(src_path, dst_path)
# pages = convert_from_path('AC010M00.02 EDs and Panel Layouts-R00.pdf')


# In[35]:


# inputpdf = PdfFileReader(open(str(pdfdocs[0]),'rb'))
# maxPages = inputpdf.numPages


# In[38]:


try:
    for l in range(0,len(pdfdocs)):
        if pdfdocs[l][0:4]=="SSRN":
            doc_name = str(pdfdocs[l])
            text = convert_pdf_to_txt(str(pdfdocs[l]))
            topic = re.split(",",re.split("\n",re.split("Abstract",re.split("Introduction",text)[0])[0].replace("\n\n"," "))[0])[0]
            if len(topic) > 100:
                topic = re.split(",",re.split("\n",re.split("Abstract",re.split("Introduction",text)[0])[0].replace("\n\n"," "))[0])[0][0:100]
            try:
                abstract = re.split("Abstract",re.split("Introduction",text)[0])[1].replace("\n\n"," ").replace("\n"," ")
            except:
                abstract = ""
            if os.path.isdir("SSRN/"+str(topic)) is False:
                os.mkdir("SSRN/"+str(topic))
            output = open("SSRN/"+str(topic)+"/"+str(doc_name)+".txt","w")
            output.write(abstract)
            output.close()
            shutil.copy(pdfdocs[l], "SSRN/"+str(topic)+"/")
except:
    pass


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





# In[ ]:





# In[11]:


# def pdf_page_to_png(src_pdf, pagenum=0, resolution=154):
#     """
#     Returns specified PDF page as wand.image.Image png.
#     :param PyPDF2.PdfFileReader src_pdf: PDF from which to take pages.
#     :param int pagenum: Page number to take.
#     :param int resolution: Resolution for resulting png in DPI.
#     """

#     check_dependencies(__optional_dependencies__['pdf'])
#     # Import libraries within this function so as to avoid import-time dependence
#     import PyPDF2
#     from wand.image import Image  # TODO: When we start using this again, document which system-level libraries are required.

#     dst_pdf = PyPDF2.PdfFileWriter()
#     dst_pdf.addPage(src_pdf.getPage(pagenum))

#     pdf_bytes = io.BytesIO()
#     dst_pdf.write(pdf_bytes)
#     pdf_bytes.seek(0)

#     img = Image(file=pdf_bytes, resolution=resolution)
#     img.convert("png")

#     return img 


# In[ ]:





# In[17]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




