import numpy as np
import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import gensim
import gensim.downloader as api
import process_data 

from spellchecker import SpellChecker

spell = SpellChecker()
spell.word_frequency.add('obama')
spell.word_frequency.add('blm')
spell.word_frequency.add('killing')


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


stop_words = set(stopwords.words('english'))
stop_words.add('')

"""
Cleans the dataset and returns the 

@param file_lines: list of lines in the input file where each line contains all the information for a given comment (content + title + author title + etc.)

@returns [labels, comment_list, title_list, max_len, max_title_len]
    labels: file
    comment_list: list of all comments in the file
    title_list: list of all titles in the file
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


"""
Returns word2vec embeddings for an input word string

@param word : a string
@param embed : the embedding keyed vectors (in our case word2vec)
@returns : the (300,0) embedding for word
"""
def get_embed(word, embed):
    x = np.zeros((300,)) # default value should be 0
    corrected = spell.correction(word) # closest correction
    if word in embed: # base word
        x = embed[word]
    elif word.upper() in embed: # capitalized (edge case for acronyms like BLM) (for some reason blm doesn't exist but BLM does?)
        x = embed[word.upper()]
    elif word.lower() in embed: # opposite of capitalization
        x = embed[word.lower()]
    elif corrected in embed: # last case, check if closest correction exists (might be bad, some corrections are kinda ass)
        x = embed[corrected]
    
    return x

"""
Converts the lists for comments, titles into ndarrays

@params : straightforward
@returns: [comment_array,title_array] list of ndarrays for comments and titles
"""
def to_array(embed, comments, titles, max_comment_len, max_title_len):
    comment_array = np.zeros((len(comments), max_comment_len, 300))
    title_array = np.zeros((len(titles), max_title_len, 300))
    for ix1, sent in enumerate(comments):
        for ix2, word in enumerate(sent):
            comment_array[ix1,ix2] = get_embed(word,embed)
    for ix1, title in enumerate(titles):
        for ix2, word in enumerate(title):
            title_array[ix1,ix2] = get_embed(word,embed)
    
    return comment_array, title_array

"""
Randomly shuffles the outputs of to_array
"""
def custom_shuffle(comments,titles,labels):
    """
    comments/title is a (batch_size, max_comment/title_length,embedding size) ndarray, this means we need batch_first=true in nn.LSTM
    comment_array.shape = (batch_size, max_comment_len, 300) #300 is wor2vec embedding size
    labels.shape = (batch_size,)
    """
    num_samples, _ , _ = comments.shape
    shuffled_indices = np.random.permutation(num_samples) #return a permutation of the indices
    
    shuffled_comments = comments[shuffled_indices,:,:]
    shuffled_titles = titles[shuffled_indices,:,:]
    shuffled_labels = labels[shuffled_indices]

    return (shuffled_comments, shuffled_titles, shuffled_labels)

"""
# expected results of verification code

print(train_max_len) # 244
print(train_max_title_len) # 13
print(len(train_text)) # 1528
print(len(train_title)) # 1528
print(train_labels.shape) # (1528,)

print(test_max_len) # 126
print(test_max_title_len) # 19
print(len(test_text)) # 102
print(len(test_title)) # 102
print(test_labels.shape) # (102,)
"""   

path = api.load("word2vec-google-news-300", return_path=True)
#print(path) #/root/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz

embed = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

train_lines = open("fox-news-comments.json", "r").readlines() #original 2015 data
test_lines = open("modern_comments.json", "r").readlines() #modern data

train_labels, train_comments, train_titles, train_max_len, train_max_title_len = process_data.clean(train_lines)
test_labels, test_comments, test_titles, test_max_len, test_max_title_len = process_data.clean(test_lines)

train_comment_array, train_title_array = to_array(embed, train_comments, train_titles, train_max_len, train_max_title_len)
test_comment_array, test_title_array = to_array(embed, test_comments, test_titles, test_max_len, test_max_title_len)

train_comment_array,train_title_array,train_labels = custom_shuffle(train_comment_array,train_title_array,train_labels)
test_comment_array, test_title_array, test_labels = custom_shuffle(test_comment_array, test_title_array, test_labels)

"""
print(train_data_array.dtype) # float64
print(type(train_data_array)) # <class 'numpy.ndarray'>
print(train_title_array.dtype) # float64
print(type(train_title_array)) # <class 'numpy.ndarray'>
"""

train_comment_array = np.float32(train_comment_array)
train_title_array = np.float32(train_title_array)

test_comment_array = np.float32(test_comment_array)
test_title_array = np.float32(test_title_array)