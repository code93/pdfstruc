#!/usr/bin/env python
# coding: utf-8

# In[154]:


import os
import shutil
# !pip install textract
# import textract
#!pip install pdf2image
from pdf2image import convert_from_path
#!pip install PyPDF3
from PyPDF3 import PdfFileWriter,PdfFileReader
from decouple import config
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import regex as re
import cv2
import numpy as np
nlp = spacy.load("en_core_web_sm")


# In[155]:


pathRel = "./"
pathAbs = ""

contents = os.listdir(pathRel)

pdfdocs = []
output_folder_paths = []
for i in range(0,len(contents)):
        if contents[i].lower().endswith(('.pdf'))==True:
            pdfdocs.append(contents[i])
            output_folder_paths.append(pathRel+str(contents[i][:-4]))


# In[156]:

if config("use_process_again")=='true':
    for j in range(0,len(pdfdocs)):
        if os.path.isdir(output_folder_paths[j]) is True:
            os.rmdir(output_folder_paths[j])
    if config("use_nlp_topic_modelling")=='true':
        if os.path.isdir("Topic_1") is True:
            os.rmdir("Topic_1")
        if os.path.isdir("Topic_2") is True:
            os.rmdir("Topic_2")
        if os.path.isdir("Topic_3") is True:
            os.rmdir("Topic_3")
        if os.path.isdir("Topic_4") is True:
            os.rmdir("Topic_4")
        if os.path.isdir("Topic_5") is True:
            os.rmdir("Topic_5")
        if os.path.isdir("Topic_6") is True:
            os.rmdir("Topic_6")
        if os.path.isdir("Topic_7") is True:
            os.rmdir("Topic_7")
    if config("use_regex_for_research_paper")=='true':
        if os.path.isdir("ResearchPapers") is True:
            os.rmdir("ResearchPapers")
        if os.path.isdir("SSRN") is True:
            os.rmdir("SSRN")

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


# In[157]:


# pdf_file_paths = []
# for pdfdoc in pdfdocs:
#     absolute = pathRel+str(pdfdoc)
#     pdf_file_paths.append(absolute)
# #     text = textract.process(absolute)
# #     files.append(text)
# len(pdfdocs)


# In[158]:


articles = []
for j in range(0,len(pdfdocs)):
    if os.path.isdir(output_folder_paths[j]) is False:
        os.mkdir(output_folder_paths[j])
        os.mkdir(str(output_folder_paths[j])+"/images")
        inputpdf = PdfFileReader(open(str(pdfdocs[j]),'rb'))
        try:
            if config("use_nlp")=='true':
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


# In[159]:


#         inputpdf = PdfFileReader(open(str(pdfdocs[j]),'rb'))
           
#             for i in range (0, inputpdf.numPages):
#                 doc.append(inputpdf.getPage(i).extractText())
#             def listToString(s): 
#                 str1 = "" 
#                 for ele in s: 
#                     str1 += ele
#                 return str1
#             text = listToString(doc)


# In[160]:


#npr['Topic']= 8


# npr = pd.DataFrame()
# npr['Articles'] = articles
# npr['Topic'] = topic_results.argmax(axis=1)
if config("use_nlp_topic_modelling")=='true':
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
    if config("use_regex_for_research_paper")=='true' and config("use_for_ssrn")=='false':
        if os.path.isdir("SSRN") is False:
            os.mkdir("ResearchPapers")
    if config("use_regex_for_research_paper")=='true' and config("use_for_ssrn")=='true':
        if os.path.isdir("SSRN") is False:
            os.mkdir("SSRN")
# os.mkdir(pathRel+str(pdfdocs[0][:-4]))


# In[161]:

if config("use_nlp_topic_modelling")=='true':
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


# In[162]:


# inputpdf = PdfFileReader(open(str(pdfdocs[0]),'rb'))
# maxPages = inputpdf.numPages


# In[163]:

if config("use_regex_for_research_paper")=='true':
    try:
        if config("use_for_ssrn")=='false':
            pdfdocs = config('pdfdocs')
            for l in range(0,len(pdfdocs)):
                doc_name = str(pdfdocs[l])
                text = convert_pdf_to_txt(str(pdfdocs[l]))
                topic = re.split(",",re.split("\n",re.split("Abstract",re.split("Introduction",text)[0])[0].replace("\n\n"," "))[0])[0]
                if len(topic) > 100:
                    topic = re.split(",",re.split("\n",re.split("Abstract",re.split("Introduction",text)[0])[0].replace("\n\n"," "))[0])[0][0:100]
                try:
                    abstract = re.split("Abstract",re.split("Introduction",text)[0])[1].replace("\n\n"," ").replace("\n"," ")
                except:
                    abstract = ""
                if os.path.isdir("ResearchPapers/"+str(topic)) is False:
                    os.mkdir("ResearchPapers/"+str(topic))
                output = open("ResearchPapers/"+str(topic)+"/"+str(doc_name)+".txt","w")
                output.write(abstract)
                output.close()
                shutil.copy(pdfdocs[l], "ResearchPapers/"+str(topic)+"/")
        if config("use_for_ssrn")=='true':
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


# In[164]:


# print(maxPages)


# In[165]:


# i = 1
# for page in range(1,maxPages,10):
#     pil_images = convert_from_path(str(pdfdocs[0]), dpi=200, first_page=page, 
#                                    last_page=min(page+10-1,maxPages),fmt='jpg', 
#                                    thread_count=1,userpw=None,use_cropbox=False, strict=False)
#     for image in pil_images:
#         image.save(str(output_folder_paths[0])+'/'+'output'+str(i)+'.jpg','JPEG')
#         i = i + 1


# In[ ]:





# In[166]:


# from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer3.converter import TextConverter
# from pdfminer3.layout import LAParams
# from pdfminer3.pdfpage import PDFPage
# from io import StringIO

# def convert_pdf_to_txt(path):
#     rsrcmgr = PDFResourceManager()
#     retstr = StringIO()
#     codec = 'utf-8'
#     laparams = LAParams()
#     device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
#     fp = open(path, 'rb')
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     password = ""
#     maxpages = 0
#     caching = True
#     pagenos=set()

#     for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
#         interpreter.process_page(page)

#     text = retstr.getvalue()

#     fp.close()
#     device.close()
#     retstr.close()
#     return text


# In[167]:


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


# In[176]:

if config("use_regex_for_page_no")=='true':
    try:
        text = convert_pdf_to_txt('AC010M00.02 EDs and Panel Layouts-R00.pdf')
        first = re.split("next page:",text.lower())
        inputpdf = PdfFileReader(open('AC010M00.02 EDs and Panel Layouts-R00.pdf','rb'))
        page_numbers = []
        for m in range(1,inputpdf.numPages):
            page_numbers.append(re.split("\n\n",re.split("page:",first[m])[1].strip())[0])
        output = open("AC010M00.02 EDs and Panel Layouts-R00/AC010M00.02 EDs and Panel Layouts-R00.txt","w")
        output.write(text)
        output.close()
        try:
            n = 1
            for page_number in page_numbers:
                filename =  "PageNo"+str(page_number)+".jpg"
                os.rename('AC010M00.02 EDs and Panel Layouts-R00/images/PageNo '+''+n+'.jpg',filename)
                n+1
        except:
            pass
    except:
        pass



# In[169]:


# text = convert_pdf_to_txt('AC010M00.02 EDs and Panel Layouts-R00.pdf')
# first = re.split("next page:",text.lower())
# text


# In[170]:


# page_numbers = []
# for m in range(1,inputpdf.numPages):
#     page_numbers.append(re.split("\n\n",re.split("page:",first[m])[1].strip())[0])


# In[171]:


# output = open("AC010M00.02 EDs and Panel Layouts-R00/AC010M00.02 EDs and Panel Layouts-R00.txt","w")
# output.write(text)
# output.close()


# In[172]:


# re.split("\n\n",re.split("page:",first[1])[1].strip())[0]


# In[173]:


#page_numbers


# In[175]:


# try:
#     n = 1
#     for page_number in page_numbers:
#         filename =  "PageNo"+str(page_number)+".jpg"
#         os.rename('AC010M00.02 EDs and Panel Layouts-R00/images/PageNo '+''+n+'.jpg',filename)
#         n+1
# except:
#     pass


# In[ ]:


# ['1000',
#  '1000.a',
#  '1003',
#  '1003.a',
#  '1003.b',
#  '1003.c',
#  '1003.d',
#  '1003.e',
#  '1003.f',
#  '1030',
#  '1040',
#  '1050',
#  '1051',
#  '1052',
#  '1053',
#  '1150',
#  '1151',
#  '1152',
#  '1153',
#  '1154',
#  '1155',
#  '1176',
#  '1413',
#  '1414',
#  '1415',
#  '1416',
#  '1417',
#  '1418',
#  '1600',
#  '1920',
#  '1921',
#  '1922',
#  '2050',
#  '2051',
#  '2176',
#  '2401',
#  '2402',
#  '2402.a',
#  '2403',
#  '2404',
#  '2405',
#  '2406',
#  '2407',
#  '2408',
#  '2409',
#  '2410',
#  '2411',
#  '2412',
#  '2413',
#  '2414',
#  '2415',
#  '2416',
#  '2600',
#  '2920',
#  '2921',
#  '2922',
#  '3050',
#  '3051',
#  '3176',
#  '3400',
#  '3400.a',
#  '3401',
#  '3402',
#  '3403',
#  '3404',
#  '3405',
#  '3406',
#  '3407',
#  '3408',
#  '3409',
#  '3410',
#  '3411',
#  '3412',
#  '3413',
#  '3414',
#  '3415',
#  '3416',
#  '3417',
#  '3600',
#  '3920',
#  '3921',
#  '3922',
#  '4050',
#  '4051',
#  '4176',
#  '4400',
#  '4401',
#  '4402',
#  '4403',
#  '4404',
#  '4405',
#  '4406',
#  '4407',
#  '4408',
#  '4409',
#  '4410',
#  '4411',
#  '4600',
#  '4920',
#  '4921',
#  '5050',
#  '5051',
#  '5052',
#  '5053',
#  '5054',
#  '5055',
#  '5056',
#  '5057',
#  '5058',
#  '5059',
#  '5060',
#  '5061',
#  '5062',
#  '5063',
#  '5064',
#  '5065',
#  '5066',
#  '5067',
#  '5068',
#  '5069',
#  '5176',
#  '5177',
#  '5178',
#  '5179',
#  '5180',
#  '5200',
#  '5201',
#  '5202',
#  '5203',
#  '5204',
#  '5205',
#  '5206',
#  '5207',
#  '5208',
#  '5209',
#  '5210',
#  '5211',
#  '5212',
#  '5213',
#  '5214',
#  '5215',
#  '5216',
#  '5217',
#  '5218',
#  '5219',
#  '5220',
#  '5221',
#  '5222',
#  '5223',
#  '5224',
#  '5225',
#  '5226',
#  '5227',
#  '5228',
#  '5229',
#  '5230',
#  '5231',
#  '5232',
#  '5233',
#  '5234',
#  '5235',
#  '5236',
#  '5237',
#  '5237.a',
#  '5237.b',
#  '5237.c',
#  '5237.d',
#  '5238',
#  '5238.a',
#  '5238.b',
#  '5239',
#  '5240',
#  '5240.a',
#  '5240.b',
#  '5240.c',
#  '5241',
#  '5242',
#  '5250',
#  '5252',
#  '5253',
#  '5254',
#  '5391',
#  '5600',
#  '8200',
#  '8201',
#  '8202',
#  '8203',
#  '8204',
#  '8205',
#  '8206',
#  '8207',
#  '8208',
#  '8209',
#  '8210',
#  '8600',
#  '8250',
#  '8251',
#  '8252',
#  '8253',
#  '8254',
#  '8255',
#  '8600',
#  '9200',
#  '9201',
#  '9202',
#  '9203',
#  '9204',
#  '9205',
#  '9206',
#  '9207',
#  '9208',
#  '9209',
#  '9210',
#  '9211',
#  '9212',
#  '9220',
#  '9221',
#  '9222',
#  '9223',
#  '9224',
#  '9225',
#  '9226',
#  '9227',
#  '9228',
#  '9229',
#  '9230',
#  '9231',
#  '9232',
#  '9233',
#  '9234',
#  '9240',
#  '9241',
#  '9242',
#  '9243',
#  '9244',
#  '9245',
#  '9246',
#  '9247',
#  '9248',
#  '9260',
#  '9261',
#  '9262',
#  '9263',
#  '9264',
#  '9265',
#  '9266',
#  '9267',
#  '9268',
#  '9269',
#  '9270',
#  '9271',
#  '9272',
#  '9273',
#  '9274',
#  '9275',
#  '9276',
#  '9277',
#  '9278',
#  '9300',
#  '9301',
#  '9302',
#  '9303',
#  '9304',
#  '9305',
#  '9306',
#  '9307']

