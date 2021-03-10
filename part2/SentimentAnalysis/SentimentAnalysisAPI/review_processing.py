import pandas as pd
import string
import tensorflow as tf
from nltk.tokenize import word_tokenize
import numpy as np

max_len=50
vocab={}
model_path = 'SentimentAnalysisAPI/model'
model = tf.keras.models.load_model(model_path)
stopwordList=[]


def load_stop_words():
    global stopwordList
    f=open('SentimentAnalysisAPI/stopwords.txt','r')
    lines=f.readlines()
    f.close()
    stopwordList = [line.rstrip() for line in lines]

def load_vocab():
    global vocab
    f = open('SentimentAnalysisAPI/vocab.csv','r')
    lines = f.readlines()
    f.close()
    for line in lines:
        li = list(line.split(","))
        vocab[li[0]] =li[1].rstrip()
    #print(vocab["like"])

def convert_to_lower(text):
    return [i.lower() for i in text]

def remove_punctuation(text):
    return [i.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) for i in text]

def encode_data(text):
    text_encoded = np.zeros((1,max_len))
    j = 0
    for w in text[0]:
        try:
            text_encoded[0, j] = vocab[w]
        except:
            pass
        j = j + 1
    return text_encoded

def remove_stopwords(text):
    newtext = []
    for tokens in text:
        newtext.append([w for w in tokens if not w in stopwordList])
    return newtext

def perform_tokenization(text):
    return [word_tokenize(i) for i in text]

def preprocess_data(text):
    review = convert_to_lower([text])
    review = remove_punctuation(review)
    review = perform_tokenization(review)
    review = remove_stopwords(review)
    review = encode_data(review)
    return review

def predict_ratings(text):
    review = preprocess_data(text)
    return model.predict(review)
