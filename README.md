## Install dependencies
### do apt install tesseract only when using ocr to do the same
### use spacy download only when using nlp (pdfstruc.py script)
```bash
apt install tesseract-ocr -y
pip install -r requirements.txt
spacy download en_core_web_sm
```

## Execute Python Script for getting images named as page numbers correctly and text data of pdf 

```bash
python3 pdf2jpg.py
```

## To do the same using ocr
### go to the directory containing images of pdf file and execute the script

```bash
python3 pdfocr.py
```


## Execute Python Script for also getting summary of pdf's and using Topic Modelling to sort with relevency:

```bash
python3 pdfstruc.py
```

###  If there is any pdf file that has not been processed to jpg images in the directory where the .py script executes, the py script automatically creates a folder of the same name of the pdf file and then starts converting pdf pages to jpg images saved with appropriate page name.


