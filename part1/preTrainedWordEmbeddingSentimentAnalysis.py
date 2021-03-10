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

tf.random.set_seed(1)
np.random.seed(1)
np.set_printoptions(threshold=np.inf)
MAX_SENTENCE_LENGTH = 40
WORD_EMBEDDING_VECTOR_SIZE=30
NUMBER_CLASSES = 5
vocab = dict()

'''
words not in train: terriblr, johnny, solicitous, needlesharp, weddings, paths, beacause etc
'''
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
        self.model = None

    def build_nn(self):
        featureDim = self.reviews.shape[1]
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(NUMBER_CLASSES, activation=softmax_activation, input_shape=(featureDim,)))
        self.model.summary()
        '''self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])'''
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                           metrics=['sparse_categorical_accuracy'])

    def train_nn(self, batch_size, epochs):
        self.model.fit(self.reviews, self.ratings - 1, epochs=epochs, batch_size=batch_size)

    def predict(self, reviews):
        reviews = np.array(reviews, dtype='float32')
        return np.argmax(self.model.predict(reviews), axis=1) + 1


def evaluation_matrices(y, y_hat):
    confusionMatrix = np.zeros((NUMBER_CLASSES+1, NUMBER_CLASSES+1))
    for i in range(len(y)):
        confusionMatrix[y[i]][y_hat[i]]+=1
    #print(confusionMatrix)
    overallTruePositive=0
    overallFalsePositive=0
    overallFalseNegative=0
    for i in range(1, NUMBER_CLASSES+1):
        overallTruePositive+=confusionMatrix[i][i]
        overallFalseNegative+=np.sum(confusionMatrix, axis=1)[i]-confusionMatrix[i][i]
        overallFalsePositive+=np.sum(confusionMatrix, axis=0)[i]-confusionMatrix[i][i]
        recall=1.0*confusionMatrix[i][i]/np.sum(confusionMatrix, axis=1)[i]
        precision = 1.0*confusionMatrix[i][i] / np.sum(confusionMatrix, axis=0)[i]
        f1score = 2.0*recall*precision/(recall+precision)
        print("Class %d (Recall: %.2f, Precision: %.2f, F1-score: %.2f )"%(i, recall, precision, f1score))
    overallRecall=1.0*overallTruePositive/(overallTruePositive+overallFalseNegative)
    overallPrecision = 1.0 * overallTruePositive / (overallTruePositive + overallFalsePositive)
    overallf1Score=2.0*overallRecall*overallPrecision/(overallRecall+overallPrecision)
    #print("overallTruePositive: %d, trace: %d"%(overallTruePositive, np.trace(confusionMatrix)))
    print("overall (accuracy: %.2f, Recall: %.2f, Precision: %.2f, F1-score: %.2f)"%(1.0*np.trace(confusionMatrix)/np.sum(confusionMatrix), overallRecall, overallPrecision, overallf1Score ))

# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    train_ratings = np.array(train_data["ratings"])
    batch_size, epochs = 128, 20

    train_reviews = preprocess_data(train_data)
    test_reviews = preprocess_data(test_data, False)

    model = NeuralNet(train_reviews, train_ratings)
    model.build_nn()
    model.train_nn(batch_size, epochs)
    print("=================Train data evaluation metrices==========================")
    evaluation_matrices(train_ratings, model.predict(train_reviews))
    print("=================Test data evaluation metrices==========================")
    testPredictions = model.predict(test_reviews)
    test_ground_truth = np.array(pd.read_csv('gold_test.csv')['ratings'])

    evaluation_matrices(test_ground_truth, testPredictions)

    return testPredictions


main("train.csv", "test.csv")
