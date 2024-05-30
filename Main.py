import pytesseract
from PIL import Image
import cv2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog
import fitz

# Set the Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  


def extract_text_from_image(imagename):
    image = cv2.imread(imagename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def preprocess_text(text):
    # Remove non-word characters and convert to lower case
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    return text

def summarize_text(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)

    freq_table = {}
    for word in words:
        word = word.lower()
        if word not in stop_words:
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

    sentences = sent_tokenize(text)
    sentence_value = {}
    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                if sentence in sentence_value:
                    sentence_value[sentence] += freq
                else:
                    sentence_value[sentence] = freq

    sum_values = sum(sentence_value.values())
    average = int(sum_values / len(sentence_value))

    summary = ''
    for sentence in sentences:
        if sentence in sentence_value and sentence_value[sentence] > 1.2 * average:
            summary += " " + sentence    
    return summary

def advanced_summarize_text(text):
    summarizer = pipeline('summarization')
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def main():
    root = tk.Tk()
    root.withdraw()
    
    # Open a file dialog to select the image file
    imagename = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    
    if imagename:
        text = extract_text_from_image(imagename)
        cleaned_text = preprocess_text(text)
        summary = advanced_summarize_text(cleaned_text)
        print("Extracted Text:\n", text)
        print("\nSummary:\n", summary)

    else:
        print("No file selected.")

main()
