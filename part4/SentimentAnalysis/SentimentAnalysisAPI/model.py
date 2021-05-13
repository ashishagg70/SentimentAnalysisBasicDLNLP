import tensorflow as tf
import numpy as np
from transformers import TFDistilBertModel,DistilBertTokenizer

max_len = 100
NUMBER_CLASSES=5
bert_model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
bert = TFDistilBertModel.from_pretrained(bert_model_name)

class Generator(tf.keras.utils.Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set,mask_set, y_set, batch_size=256):

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
        
        return [batch_x,batch_mask] ,batch_y
    

class NeuralNet:

    def __init__(self):
        self.model = None

    def build_nn(self,hidden_layer_count,hidden_layer_numunits):
        sentence_indices = tf.keras.layers.Input(shape=(max_len,),dtype='int32')
        attention_masks = tf.keras.layers.Input(shape=(max_len,),dtype='int32')
        X = bert(sentence_indices,attention_mask = attention_masks)[0]
        # X = tf.keras.layers.GlobalAveragePooling1D()(X)
        X = tf.keras.layers.Flatten()(X)
    
        for i in range(hidden_layer_count):
            X = tf.keras.layers.Dense(units=hidden_layer_numunits[i],activation = 'relu')(X)
            X = tf.keras.layers.Dropout(rate=0.2)(X)
        
        X = tf.keras.layers.Dense(units=5)(X)

        X = tf.keras.activations.softmax(X,axis =1)
        
        self.model = tf.keras.Model(inputs=[sentence_indices,attention_masks],outputs=X)
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), metrics=['accuracy'])
        self.model.summary()
    
    def load_weights(self,path):
        self.model.load_weights(path)

    def train_nn(self, batch_size, epochs, reviews, attention_masks, ratings, val_reviews, val_attention_masks, val_ratings):
        self.reviews = reviews
        self.attention_masks = attention_masks
        self.ratings = tf.keras.utils.to_categorical(y=ratings-1,num_classes=NUMBER_CLASSES)
        val_ratings = tf.keras.utils.to_categorical(y=val_ratings-1,num_classes=NUMBER_CLASSES)
        self.val_data=([val_reviews,val_attention_masks],val_ratings)
        self.model.fit(Generator(self.reviews,self.attention_masks,self.ratings,batch_size), epochs=epochs, batch_size=batch_size,validation_data=self.val_data,steps_per_epoch = 300)
        
    def predict(self, reviews):
        reviews = np.array(reviews, dtype='float32')
        return np.argmax(self.model.predict(reviews), axis=1) + 1
    
    def predictWithPr(self, reviews,attention_masks):
        reviews = np.array(reviews)
        attention_masks = np.array(attention_masks)
        return self.model.predict([reviews,attention_masks])