import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from collections import defaultdict
import math
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.random.set_seed(1)
np.random.seed(1)
np.set_printoptions(threshold=np.inf)

max_len=50
NUMBER_CLASSES = 5
EMBDIM=300
hidden_layer_count=0
hidden_layer_numunits=[256]

fastText_file='./cc.en.300.vec'

word_to_index = defaultdict(int)

index_to_word = defaultdict(str)

filepath = './SentimentAnalysisData/'

embeddings={}

def pretrain_embedding():
    global embeddings
    f=open(fastText_file, 'r')
    lines=f.readlines()
    f.close()
    for line in lines[1:]:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings[word] = vector

def train_embedding(data, create_new=False):
    global embeddings
    if create_new == True:
        reviews_mod = []
        for sent in data:
            reviews_mod.append(list(map(lambda x:index_to_word[int(x)],sent)))
        model = gensim.models.Word2Vec(reviews_mod, min_count=1,
                                       size=EMBDIM, window=5)
        model.wv.save("train_word2vec.wordvectors")
    embeddings = KeyedVectors.load("train_word2vec.wordvectors", mmap='r')

def convert_to_lower(text):
    return [i.lower() for i in text]

def remove_punctuation(text):
    return [i.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) for i in text]

def encode_data(text,isTrain=True):
    global word_to_index
    global index_to_word
    if isTrain:
        f = open(filepath+'vocab.csv','r')
        lines = f.readlines()
        f.close()
        for line in lines:
            li = list(line.split(","))
            index= int(li[1].rstrip())
            word_to_index[li[0]] =index
            index_to_word[index] = li[0]
    m=len(text)
    text_encoded = np.zeros((m,max_len))
    for i in range(m):
        sentence_words = text[i]
        j = 0
        for w in sentence_words:
            text_encoded[i, j] = word_to_index.get(w)
            j = j + 1
    return text_encoded

def remove_stopwords(text):
    newtext = []
    f=open(filepath+'stopwords.txt','r')
    lines=f.readlines()
    stopwordList = [line.rstrip() for line in lines]
    for tokens in text:
        newtext.append([w for w in tokens if not w in stopwordList])
    return newtext

def perform_tokenization(text):
    return [word_tokenize(i) for i in text]

def preprocess_data(data, isTrain=True):
    review = data["reviews"]
    review = convert_to_lower(review)
    review = remove_punctuation(review)
    review = perform_tokenization(review)
    review = remove_stopwords(review)
    review = encode_data(review,isTrain)    
    return review

def embedding_layer():
    
    vocab_size = len(word_to_index) + 1           
    emb_dim = EMBDIM
    emb_matrix = np.zeros((vocab_size,emb_dim))
    for word, idx in word_to_index.items():
        try:
            emb_matrix[idx, :] = embeddings[word]
        except:
            pass
    
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=emb_dim,weights=[emb_matrix],input_length=max_len,trainable=True)
    
    return embedding_layer

def softmax_activation(x):
    expX = K.exp(x-K.reshape(K.max(x, axis=1), (K.shape(x)[0], 1)))
    s = K.reshape(K.sum(expX, axis=1), (K.shape(x)[0], 1))
    return expX / s    



class Generator(tf.keras.utils.Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        y_temp=np.argmax(y_set,axis=1)+1
        self.batch_size = batch_size
    
        self.c1_review = x_set[y_temp==1,:]
        self.c2_review = x_set[y_temp==2,:]
        self.c3_review = x_set[y_temp==3,:]
        self.c4_review = x_set[y_temp==4,:]
        self.c5_review = x_set[y_temp==5,:]
        
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
        self.batch_1=int(batch_size*0.199)
        self.batch_2=int(batch_size*0.199)
        self.batch_3=int(batch_size*0.199)
        self.batch_4=int(batch_size*0.199)
        self.batch_5=self.batch_size-self.batch_1-self.batch_2-self.batch_3-self.batch_4

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        idx1=np.random.choice(np.arange(self.len1),replace=False,size=self.batch_1)
        idx2=np.random.choice(np.arange(self.len2),replace=False,size=self.batch_2)
        idx3=np.random.choice(np.arange(self.len3),replace=False,size=self.batch_3)
        idx4=np.random.choice(np.arange(self.len4),replace=False,size=self.batch_4)
        idx5=np.random.choice(np.arange(self.len5),replace=False,size=self.batch_5)
        
        batch_x=[]
        batch_x.extend(self.c1_review[idx1,:])
        batch_x.extend(self.c2_review[idx2,:])
        batch_x.extend(self.c3_review[idx3,:])
        batch_x.extend(self.c4_review[idx4,:])
        batch_x.extend(self.c5_review[idx5,:])
        batch_x=np.array(batch_x)
        
        batch_y=[]
        batch_y.extend(self.c1_ratings[idx1,:])
        batch_y.extend(self.c2_ratings[idx2,:])
        batch_y.extend(self.c3_ratings[idx3,:])
        batch_y.extend(self.c4_ratings[idx4,:])
        batch_y.extend(self.c5_ratings[idx5,:])
        batch_y=np.array(batch_y)
        
        return batch_x, batch_y
    



class NeuralNet:

    def __init__(self, reviews, ratings, val_reviews,val_ratings):
        self.reviews = np.array(reviews, dtype='float32')
        self.ratings = tf.keras.utils.to_categorical(y=ratings-1,num_classes=NUMBER_CLASSES)
        val_ratings = tf.keras.utils.to_categorical(y=val_ratings-1,num_classes=NUMBER_CLASSES)
        self.val_data=(val_reviews,val_ratings)
        self.model = None

    def build_nn(self,hidden_layer_count,hidden_layer_numunits, rnn_type, num_rnn):
        
        sentence_indices = tf.keras.layers.Input(shape=(max_len,),dtype='int32')
        embedding = embedding_layer()
        X = embedding(sentence_indices)
        #X = tf.keras.layers.Flatten()(X)
        if rnn_type == "rnn":
            for i in range(num_rnn-1):
                X = tf.keras.layers.SimpleRNN(256,return_sequences=True,dropout=0.2)(X)
            X = tf.keras.layers.SimpleRNN(256)(X)
        elif rnn_type == "lstm":
            for i in range(num_rnn-1):
                X = tf.keras.layers.LSTM(256,return_sequences=True,dropout=0.2)(X)
            X = tf.keras.layers.LSTM(256)(X)
        elif rnn_type == "bilstm":
            for i in range(num_rnn-1):
                X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,return_sequences=True,dropout=0.2))(X)
            X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,dropout=0.2))(X)
        elif rnn_type == "gru":
            for i in range(num_rnn-1):
                X = tf.keras.layers.GRU(256,return_sequences=True,dropout=0.2)(X)
            X = tf.keras.layers.GRU(256,dropout=0.2)(X)
        elif rnn_type == "bigru":
            for i in range(num_rnn-1):
                X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256,return_sequences=True,dropout=0.2))(X)
            X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256,dropout=0.2))(X)

        for i in range(hidden_layer_count):
            X = tf.keras.layers.Dense(units=hidden_layer_numunits[i],kernel_initializer='glorot_uniform')(X)
            X = tf.keras.layers.Dropout(rate=0.2)(X)
            X=tf.keras.layers.Activation(activation='relu')(X)
        
        X = tf.keras.layers.Dense(units=5,activation='softmax')(X)
        
        self.model = tf.keras.Model(inputs=sentence_indices,outputs=X)
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
        self.model.summary()

    def train_nn(self, batch_size, epochs):
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1,patience=5)
        self.model.fit(Generator(self.reviews,self.ratings,batch_size), epochs=epochs, batch_size=batch_size,validation_data=self.val_data,callbacks=[es],steps_per_epoch=40)
   
    def predict(self, reviews):
        reviews = np.array(reviews, dtype='float32')
        return np.argmax(self.model.predict(reviews), axis=1) + 1
    
    def predictWithPr(self, reviews):
        reviews = np.array(reviews, dtype='float32')
        return self.model.predict(reviews)


if __name__ == "__main__":

    epochs=int(sys.argv[1])
    batch_size=int(sys.argv[2])
    num_hidden_layer=int(sys.argv[3])
    embedding_type = sys.argv[4]
    rnn_type = sys.argv[5]
    num_rnn_layers = int(sys.argv[6])

    train_data = pd.read_csv(filepath+"train.csv")
    test_data = pd.read_csv(filepath+"test.csv")
    train_df, val_df = train_test_split(train_data, test_size=0.2)

    train_ratings = np.array(train_df["ratings"])
    train_reviews = np.array(preprocess_data(train_df))

    val_ratings = np.array(val_df["ratings"])
    val_reviews = np.array(preprocess_data(val_df, False))

    test_reviews = preprocess_data(test_data, False)

    if embedding_type == 'pretrained':
        pretrain_embedding()
    else:
        train_embedding(train_reviews,False)

    model = NeuralNet(train_reviews, train_ratings, val_reviews,val_ratings)

    model.build_nn(num_hidden_layer,[256]*num_hidden_layer, rnn_type, num_rnn_layers)

    model.train_nn(batch_size, epochs)

    print("=================Train data evaluation metrices==========================")
    trainPredictions = model.predict(train_reviews)
    print(classification_report(train_ratings,trainPredictions))
    print(precision_recall_fscore_support(train_ratings, trainPredictions,average='weighted'))

    print("=================Test data evaluation metrices==========================")

    testPredictions = model.predict(test_reviews)
    test_ground_truth = np.array(pd.read_csv(filepath+'gold_test.csv')['ratings'])

    print(classification_report(test_ground_truth, testPredictions))
    print(precision_recall_fscore_support(test_ground_truth, testPredictions,average='weighted'))
    print('Confusion Matrix:')
    print(confusion_matrix(test_ground_truth, testPredictions))

    ip_data = pd.read_csv("input.csv")
    ip_reviews = preprocess_data(ip_data, False)
    pred = model.predictWithPr(ip_reviews)
    pred = np.around(pred,3)
    print(pred)

    ip_data["rating"]=np.argmax(pred, axis=1) + 1
    print(ip_data)

