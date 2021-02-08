import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gensim
from tensorflow.keras import backend as K
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.text import Tokenizer

tf.random.set_seed(1)
np.random.seed(1)
np.set_printoptions(threshold=np.inf)
MAX_SENTENCE_LENGTH = 40
WORD_EMBEDDING_VECTOR_SIZE=30
vocab = dict()

def encode_data(text, isTrain=True):
    encodedText = []
    global vocab
    vocabLen=len(vocab)
    for sentence in text:
        encodedSentence = []
        for word in sentence:
            if word not in vocab:
                if (isTrain):
                    vocabLen += 1
                    vocab[word] = vocabLen
                else:
                    vocab[word]=0

            encodedSentence.append(str(vocab[word]))
        encodedText.append(encodedSentence)
    #print('----------', len(vocab), vocabLen, '-------')
    return encodedText


def convert_to_lower(text):
    return [i.lower() for i in text]


def remove_punctuation(text):
    return [i.translate(str.maketrans(dict.fromkeys(string.punctuation))) for i in text]

def decontract_negative_word(text):
    return [ re.sub(r"n\'t", " not", sent) for sent in text]

def remove_stopwords(text):
    newtext = []
    stopwordList = (stopwords.words('english'))
    #stopwordList=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma']
    stopwordList=set(remove_punctuation(stopwordList))
    #print('-----', stopwordList, '---------')
    for tokens in text:
        newtext.append([w for w in tokens if not w in stopwordList])
    #print(newtext[0])
    return newtext


def perform_tokenization(text):
    return [word_tokenize(i) for i in text]


def perform_padding(data):
    return [list(np.pad(sent, (0, MAX_SENTENCE_LENGTH - len(sent)), 'constant', constant_values='0')) for sent in data]
    #return np.array(pad_sequences(data, padding='post', maxlen=MAX_SENTENCE_LENGTH))

def word_embedding(data, trainEmbedding=False):
    #print(data)
    if(trainEmbedding):
        model = gensim.models.Word2Vec(data, min_count=1,
                                       size=WORD_EMBEDDING_VECTOR_SIZE, window=5)
        model.wv.save("train_word2vec.wordvectors")
    wv = KeyedVectors.load("train_word2vec.wordvectors", mmap='r')
    newText=[]
    for sent in data:
        newSent=[]
        for word in sent:
            newSent.extend(wv[word])
        newText.append(newSent)

    #print(len(wv.vocab))
    #print(wv['beacause'])
    #print(len(newText[1]))
    #print(newText[1])
    return newText

def preprocess_data(data, isTrain=True):
    review = data["reviews"]
    review = convert_to_lower(review)
    review = remove_punctuation(review)
    review = perform_tokenization(review)
    review = remove_stopwords(review)
    review = encode_data(review, isTrain)
    review = perform_padding(review)
    review = word_embedding(review, False)
    return review

def softmax_activation(x):
    expX = K.exp(x-K.reshape(K.max(x, axis=1), (K.shape(x)[0], 1)))
    #expX = np.exp(x - np.reshape(np.max(x, axis=1), (x.shape[0], 1)))
    s = K.reshape(K.sum(expX, axis=1), (K.shape(x)[0], 1))
    return expX / s

class NeuralNet:

    def __init__(self, reviews, ratings):
        self.reviews = np.array(reviews, dtype='float32')
        self.ratings = ratings
        self.numOfOutputClasses = 5
        self.model = None
        self.learningRate = 0.001
        self.W = None
        self.b = None

    def build_nn(self):
        featureDim = self.reviews.shape[1]
        self.W = tf.get_variable("W", [featureDim, self.numOfOutputClasses], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        self.b = tf.get_variable("b", [1, self.numOfOutputClasses], initializer=tf.zeros_initializer())


        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(len(vocab)+2, 30, input_length=featureDim))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.numOfOutputClasses, activation='softmax'))
        #self.model.add(tf.keras.layers.Dense(self.numOfOutputClasses, activation='softmax', input_shape=(featureDim,)))
        # model.summary()
        '''self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])'''
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                           metrics=['sparse_categorical_accuracy'])
        # add the input and output layer here; you can use either tensorflow or pytorch

    def train_nn(self, batch_size, epochs):
        init = tf.global_variables_initializer()
        sess=tf.Session()
        labels = sess.run(tf.one_hot(self.ratings - 1, self.numOfOutputClasses))
        sess.close()
        print(labels)
        X = tf.placeholder(tf.float32, [None, MAX_SENTENCE_LENGTH])
        Y = tf.placeholder(tf.float32, [None, self.numOfOutputClasses])
        Z = tf.add(tf.matmul(X, self.W), self.b)
        sZ=softmax_activation(Z)
        cost=-1*np.sum(np.log(np.sum(softmax_activation(Z)*Y, axis=1)))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(cost)
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                epochCost = 0.
                shuffle_input_ind = np.random.choice(self.reviews.shape[0], self.reviews.shape[0], replace=False)
                shuffled_train_input, shuffled_train_target = self.reviews[shuffle_input_ind], labels[
                    shuffle_input_ind]
                for i in range(0, self.reviews.shape[0], batch_size):
                    sZZ, _, minibatch_cost, sZZ = sess.run([sZ, optimizer, cost, sZ], feed_dict={X: shuffled_train_input[i:i + batch_size], Y: shuffled_train_target[i:i + batch_size]})
                    epochCost += minibatch_cost / batch_size
                print("Cost after epoch %i: %f" % (epoch, epochCost))
            self.W=sess.run(self.W)
            self.b=sess.run(self.b)
            print(self.W)
            print(self.b)
            correctPrediction = tf.equal(tf.argmax(Z,1), tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: self.reviews, Y: labels}))

    def predict(self, reviews):
        reviews = np.array(reviews, dtype='float32')
        return np.argmax(self.model.predict(reviews), axis=1) + 1
        '''return np.argmax(reviews.dot(self.W)+self.b, axis=1)+1'''
        # return a list containing all the ratings predicted by the trained model


# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    train_ratings = np.array(train_data["ratings"])
    batch_size, epochs = 128, 50

    train_reviews = preprocess_data(train_data)
    test_reviews = preprocess_data(test_data, False)

    model = NeuralNet(train_reviews, train_ratings)
    model.build_nn()
    model.train_nn(batch_size, epochs)

    testPredictions = model.predict(test_reviews)
    print(testPredictions)
    return testPredictions


main("train.csv", "test.csv")
