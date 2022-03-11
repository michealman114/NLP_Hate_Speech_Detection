import numpy as np
import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from spellchecker import SpellChecker



nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


stop_words = set(stopwords.words('english'))
stop_words.add('')

"""
Clean the dataset by

@param file_lines: list of lines in the input file

@returns [labels, comment_list, title_list, max_len, max_title_len]
    labels:
    comment_list: 
    title_list: 
    max_comment_len: length of the longest comment in the dataset
    max_title_len: length of the longest title in the dataset
"""
def clean(file_lines):
    max_len = 0
    max_title_len = 0  
    comment_list = []
    title_list = []
    label = []
    for line in file_lines:
        comment = json.loads(line)
        
        t = comment['text']
        t = ' '.join([x for x in t.split() if x[0] != '@'])
        t = ' '.join(re.findall("[a-zA-Z,.]+",t))
        t = t.replace(',', ' ')
        t = t.replace('.', ' ')
        text = word_tokenize(t)
        text = [x for x in text if x.lower() not in stop_words]
        max_len = max(max_len, len(text))
        comment_list.append(text)
        
        title = comment['title']
        title = title.replace(',', '')
        title = title.replace('.', '')
        title = re.findall("[a-zA-Z,.]+",title)
        title_list.append(title)
        max_title_len = max(max_title_len, len(title))
        
        label.append(comment['label'])
    
    labels = np.array(label)
    return labels, comment_list, title_list, max_len, max_title_len


train_lines = open("fox-news-comments.json", "r").readlines()
test_lines = open("modern_comments.json", "r").readlines()

train_labels, train_text, train_title, train_max_len, train_max_title_len = clean(train_lines)
test_labels, test_text, test_title, test_max_len, test_max_title_len = clean(test_lines)

print(clean(test_lines))
