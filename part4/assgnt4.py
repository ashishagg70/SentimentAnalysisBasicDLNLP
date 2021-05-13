import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from collections import defaultdict
import math
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import os
from transformers import DistilBertTokenizer,TFDistilBertModel
from lime.lime_text import LimeTextExplainer


nltk.download('punkt')

tf.random.set_seed(1)
np.random.seed(1)
np.set_printoptions(threshold=np.inf)

batch_size, epochs = 80, 20
max_len=100
NUMBER_CLASSES = 5
EMBDIM=300
hidden_layer_count=1
hidden_layer_numunits=[256]
word_to_index = defaultdict(int)
index_to_word = defaultdict(str)
bert_model_name = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
bert = TFDistilBertModel.from_pretrained(bert_model_name)


def remove_stopwords(text):
    newtext = []
    f=open('stopwords.txt','r')
    lines=f.readlines()
    stopwordList = [line.rstrip() for line in lines]
    for tokens in text:
        newtext.append([w for w in tokens if not w in stopwordList])
    return newtext

def perform_tokenization(text):
    return [word_tokenize(i) for i in text]

def convert_to_lower(text):
    return [i.lower() for i in text]

def remove_punctuation(text):
    return [i.translate(str.maketrans(string.punctuation.replace("'",''), ' '*len(string.punctuation.replace("'",'')))) for i in text]


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


def join_multiple(text):
    text_strings=[]
    for sent in text:
        text_strings.append(' '.join(sent))
    return text_strings


def preprocess_data(review):
    review = convert_to_lower(review)
    review = remove_punctuation(review)
    review = perform_tokenization(review)
    review = remove_stopwords(review)
    review = join_multiple(review)
    review,attention_mask = encode_data(review)  
    return review,attention_mask


class Generator(tf.keras.utils.Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set,mask_set, y_set, batch_size=256):

        print(x_set.shape)
        print(mask_set.shape)
        print(y_set.shape)
        self.x, self.y = x_set, y_set
        y_temp=np.argmax(y_set,axis=1)+1
        self.batch_size = batch_size
        
        
            
    
        self.c1_review = x_set[y_temp==1,:]
        self.c2_review = x_set[y_temp==2,:]
        self.c3_review = x_set[y_temp==3,:]
        self.c4_review = x_set[y_temp==4,:]
        self.c5_review = x_set[y_temp==5,:]

        self.c1_mask = mask_set[y_temp==1,:]
        self.c2_mask = mask_set[y_temp==2,:]
        self.c3_mask = mask_set[y_temp==3,:]
        self.c4_mask = mask_set[y_temp==4,:]
        self.c5_mask = mask_set[y_temp==5,:]
        
        
        
        self.len1=len(self.c1_review)
        self.len2=len(self.c2_review)
        self.len3=len(self.c3_review)
        self.len4=len(self.c4_review)
        self.len5=len(self.c5_review)

        self.c1_ratings = y_set[y_temp==1,:]
        self.c2_ratings = y_set[y_temp==2,:]
        self.c3_ratings = y_set[y_temp==3,:]
        self.c4_ratings = y_set[y_temp==4,:]
        self.c5_ratings = y_set[y_temp==5,:]
        self.batch_1=int(batch_size*0.2)
        self.batch_2=int(batch_size*0.2)
        self.batch_3=int(batch_size*0.2)
        self.batch_4=int(batch_size*0.2)
        self.batch_5=self.batch_size-self.batch_1-self.batch_2-self.batch_3-self.batch_4

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x=[]
        batch_mask = []
        batch_y=[]

        idx1=np.random.choice(np.arange(self.len1),replace=False,size=self.batch_1)
        batch_x.extend(self.c1_review[idx1,:])
        batch_mask.extend(self.c1_mask[idx1,:])
        batch_y.extend(self.c1_ratings[idx1,:])

        idx2=np.random.choice(np.arange(self.len2),replace=False,size=self.batch_2)
        batch_x.extend(self.c2_review[idx2,:])
        batch_mask.extend(self.c2_mask[idx2,:])
        batch_y.extend(self.c2_ratings[idx2,:])

        idx3=np.random.choice(np.arange(self.len3),replace=False,size=self.batch_3)
        batch_x.extend(self.c3_review[idx3,:])
        batch_mask.extend(self.c3_mask[idx3,:])
        batch_y.extend(self.c3_ratings[idx3,:])

        idx4=np.random.choice(np.arange(self.len4),replace=False,size=self.batch_4)
        batch_x.extend(self.c4_review[idx4,:])
        batch_mask.extend(self.c4_mask[idx4,:])
        batch_y.extend(self.c4_ratings[idx4,:])

        idx5=np.random.choice(np.arange(self.len5),replace=False,size=self.batch_5)        
        batch_x.extend(self.c5_review[idx5,:])
        batch_mask.extend(self.c5_mask[idx5,:])
        batch_y.extend(self.c5_ratings[idx5,:])

        batch_x=np.array(batch_x)
        batch_mask=np.array(batch_mask)
        
        batch_y=np.array(batch_y)

        r = np.random.permutation(self.batch_size)
        batch_x = batch_x[r]
        batch_mask = batch_mask[r]
        batch_y = batch_y[r]
        # print("b {} {} {} {}".format(idx, batch_x.shape, batch_mask.shape,batch_y.shape))
        
        return [batch_x,batch_mask] ,batch_y
    

class NeuralNet:

    def __init__(self, reviews, attention_masks, ratings, val_reviews, val_attention_masks, val_ratings):
        self.reviews = reviews
        self.attention_masks = attention_masks
        self.ratings = tf.keras.utils.to_categorical(y=ratings-1,num_classes=NUMBER_CLASSES)
        val_ratings = tf.keras.utils.to_categorical(y=val_ratings-1,num_classes=NUMBER_CLASSES)
        self.val_data=([val_reviews,val_attention_masks],val_ratings)
        self.model = None

    def build_nn(self,hidden_layer_count,hidden_layer_numunits):
        sentence_indices = tf.keras.layers.Input(shape=(max_len,),dtype='int32')
        attention_masks = tf.keras.layers.Input(shape=(max_len,),dtype='int32')
        X = bert(sentence_indices,attention_mask = attention_masks)[0]

        X = tf.keras.layers.Flatten()(X)
    
        for i in range(hidden_layer_count):
            X = tf.keras.layers.Dense(units=hidden_layer_numunits[i],activation = 'relu')(X)
            X = tf.keras.layers.Dropout(rate=0.2)(X)
        
        X = tf.keras.layers.Dense(units=5)(X)

        X = tf.keras.activations.softmax(X,axis =1)
        
        self.model = tf.keras.Model(inputs=[sentence_indices,attention_masks],outputs=X)
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), metrics=['accuracy'])
        self.model.summary()

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def train_nn(self, batch_size, epochs, balanced):
        
        if balanced == True:
          self.model.fit(Generator(self.reviews,self.attention_masks,self.ratings,batch_size), epochs=epochs, batch_size=batch_size,validation_data=self.val_data,steps_per_epoch = 250)
        else:
          self.model.fit([self.reviews,self.attention_masks],self.ratings, epochs=epochs, batch_size=batch_size,validation_data=self.val_data)
   
    def predict(self, reviews):
        reviews = np.array(reviews, dtype='float32')
        return np.argmax(self.model.predict(reviews), axis=1) + 1
    
    def predictWithPr(self, reviews):
        reviews = np.array(reviews, dtype='float32')
        return self.model.predict(reviews)

def test_for_sents(sents):
	inp_data = pd.DataFrame({'reviews':sents})
	inp_reviews,inp_mask = preprocess_data(inp_data['reviews'])
	inp_reviews,inp_mask = np.array(inp_reviews),np.array(inp_mask)
	pred = np.around(model.model.predict([inp_reviews,inp_mask]),3)
	return pred
    
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_data, test_size=0.2)

train_ratings = np.array(train_df["ratings"])
train_reviews,train_attention_masks = np.array(preprocess_data(train_df["reviews"]))
val_ratings = np.array(val_df["ratings"])
val_reviews,val_attention_masks = np.array(preprocess_data(val_df["reviews"]))
test_reviews,test_attention_masks = preprocess_data(test_data["reviews"])

model = NeuralNet(train_reviews, train_attention_masks, train_ratings, val_reviews,val_attention_masks, val_ratings)
model.build_nn(hidden_layer_count,hidden_layer_numunits)

model.train_nn(batch_size, epochs,balanced=True)

from sklearn.metrics import classification_report,confusion_matrix

print("=================Train data evaluation metrices==========================")
# evaluation_matrices(train_ratings, model.predict(train_reviews))

prediction = np.argmax(model.model.predict([train_reviews,train_attention_masks]), axis=1) + 1
print(confusion_matrix(train_ratings,prediction))
print(classification_report(train_ratings,prediction))
print(precision_recall_fscore_support(train_ratings,prediction,average='weighted'))

print("=================Test data evaluation metrices==========================")

testPredictions =np.argmax(model.model.predict([np.array(test_reviews),np.array(test_attention_masks)]), axis=1) + 1
test_ground_truth = np.array(pd.read_csv('gold_test.csv')['ratings'])

print(confusion_matrix(test_ground_truth,testPredictions))
print(classification_report(test_ground_truth,testPredictions))
print(precision_recall_fscore_support(test_ground_truth,testPredictions,average='weighted'))

explainer = LimeTextExplainer()
exp = explainer.explain_instance(sent, test_for_sents,labels=(0,1,2,3,4))
html = exp.as_html(labels=(0,1,2,3,4))
f = open('exp.html', 'w')
f.write(html)
f.close()
