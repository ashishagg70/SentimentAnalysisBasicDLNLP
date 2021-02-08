import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import numpy as np

tf.random.set_seed(1)
np.random.seed(1)
np.set_printoptions(threshold=np.inf)
MAX_SENTENCE_LENGTH = 50
WORD_EMBEDDING_VECTOR_SIZE=30
vocab = dict()

'''
results achieved: loss: 0.0030 - sparse_categorical_accuracy: 0.9997
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

            encodedSentence.append(vocab[word])
        encodedText.append(encodedSentence)
    print('----------', len(vocab), vocabLen, '-------')
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
    print('-----', stopwordList, '---------')
    for tokens in text:
        newtext.append([w for w in tokens if not w in stopwordList])
    #print(newtext[0])
    return newtext


def perform_tokenization(text):
    return [word_tokenize(i) for i in text]


def perform_padding(data):
    return np.array([np.pad(sent, (0, MAX_SENTENCE_LENGTH - len(sent)), 'constant', constant_values=0) for sent in data])
    #return np.array(pad_sequences(data, padding='post', maxlen=MAX_SENTENCE_LENGTH))


def preprocess_data(data, isTrain=True):
    review = data["reviews"]
    review = convert_to_lower(review)
    #review = decontract_negative_word(review)
    review = remove_punctuation(review)
    review = perform_tokenization(review)
    review = remove_stopwords(review)
    review = encode_data(review, isTrain)
    review = perform_padding(review)
    return review

class NeuralNet:

    def __init__(self, reviews, ratings):
        self.reviews = np.array(reviews, dtype='float32')
        self.ratings = ratings
        self.numOfOutputClasses = 5
        self.model = None

    def build_nn(self):
        featureDim = self.reviews.shape[1]
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(len(vocab)+1, WORD_EMBEDDING_VECTOR_SIZE, input_length=featureDim))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.numOfOutputClasses, activation='softmax'))
        #self.model.add(tf.keras.layers.Dense(self.numOfOutputClasses, activation='softmax', input_shape=(featureDim,)))
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

    testPredictions = np.array(model.predict(test_reviews))
    trainPredictions = np.array(model.predict(train_reviews))
    test_ground_truth=np.array(pd.read_csv('gold_test.csv')['ratings'])
    #print(testPredictions)
    #print(test_ground_truth)
    print('==test accuracy: ',np.sum(test_ground_truth==testPredictions)/test_ground_truth.shape[0],'==')
    return testPredictions


main("train.csv", "test.csv")
