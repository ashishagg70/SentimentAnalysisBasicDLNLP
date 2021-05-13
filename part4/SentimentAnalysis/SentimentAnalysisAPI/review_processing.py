import pandas as pd
import string
import tensorflow as tf
from nltk.tokenize import word_tokenize
import numpy as np
from .model import NeuralNet, tokenizer, max_len
from lime.lime_text import LimeTextExplainer
from transformers import DistilBertTokenizer

explainer = LimeTextExplainer(class_names=["1","2","3","4","5"])

model_path = 'SentimentAnalysisAPI/model/model_67'
stopwordList=[]

html_index=0

def load_model():
    global model
    model = NeuralNet()
    model.build_nn(1, [256])
    model.load_weights(model_path)

def load_stop_words():
    global stopwordList
    f=open('SentimentAnalysisAPI/stopwords.txt','r')
    lines=f.readlines()
    f.close()
    stopwordList = [line.rstrip() for line in lines]

def convert_to_lower(text):
    return [i.lower() for i in text]

def remove_punctuation(text):
    return [i.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) for i in text]

def encode_data(data):
    reviews = []
    attention = []
    for review in data:
        
        encoding =  tokenizer.encode_plus(
            review,
            max_length=max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        reviews.append(encoding["input_ids"])
        attention.append(encoding['attention_mask'])
    return reviews, attention

def remove_stopwords(text):
    newtext = []
    for tokens in text:
        newtext.append([w for w in tokens if not w in stopwordList])
    return newtext

def perform_tokenization(text):
    return [word_tokenize(i) for i in text]

def join_multiple(text):
    text_strings=[]
    for sent in text:
        text_strings.append(' '.join(sent))
    return text_strings

def preprocess_data(text):
    review = convert_to_lower(text)
    review = remove_punctuation(review)
    review = perform_tokenization(review)
    review = remove_stopwords(review)
    review = join_multiple(review)
    review,attention_mask = encode_data(review)  
    return review,attention_mask

def predict_for_sents(sents):
  inp_reviews,inp_mask = preprocess_data(sents)
  pred = np.around(model.predictWithPr(inp_reviews,inp_mask),3)
  return pred

def predict_ratings(text, num_samples):
    print(text)
    prediction = predict_for_sents([text])
    print('Predictions', prediction)
    max = np.argmax(prediction)
    print('max', max)
    exp = explainer.explain_instance(text,predict_for_sents,labels=(0,1,2,3,4),num_samples = num_samples)
    html = exp.as_html(labels=(0,1,2,3,4))
    filename = "SentimentAnalysisAPI/explains/"+str(html_index) + ".html"
    f = open(filename, 'w')
    f.write(html)
    print('html done')
    #html=None
    return prediction,filename
