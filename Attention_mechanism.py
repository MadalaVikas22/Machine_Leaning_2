# Import necessary libraries
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import tensorflow
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras import Model
import tensorflow.keras as keras
from keras.layers import Layer
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

# Reading text file
yelp_df=pd.read_csv('/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/Assignments/sentiment labelled sentences/yelp_labelled.txt',header=None, names=["Text","Sentiment"],delimiter='\t')
amazon_df=pd.read_csv('/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/Assignments/sentiment labelled sentences/amazon_cells_labelled.txt',header=None, names=["Text","Sentiment"],delimiter='\t')
# imdb_df=pd.read_csv('/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/Assignments/sentiment labelled sentences/imdb_labelled.txt',header=None, names=["Text","Sentiment"],delimiter='\t')


yelp_df.head()
amazon_df.head()

# Combining datasets
df=pd.concat([amazon_df,yelp_df])
df.reset_index(drop='True',inplace=True)
df.head()

corpus = df['Text']
t=Tokenizer()
t.fit_on_texts(corpus)
text_matrix=t.texts_to_sequences(corpus)
y = df['Sentiment']
len_mat=[]
for i in range(len(text_matrix)):
    len_mat.append(len(text_matrix[i]))

text_pad = pad_sequences(text_matrix, maxlen=32, padding='post')

# LSTM Model
inputs1 = Input(shape=(32,))
x1 = Embedding(input_dim=32 + 1, output_dim=32, input_length=32, embeddings_regularizer=keras.regularizers.l2(0.001))(inputs1)
x1 = LSTM(100, dropout=0.3, recurrent_dropout=0.2)(x1)
outputs1 = Dense(1, activation='sigmoid')(x1)
model1 = Model(inputs1, outputs1)

model1.summary()
# Compile the model
model1.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.legacy.Adam(),
              metrics=['accuracy'])

train_x, test_x, train_y, test_y = train_test_split(text_pad, np.array(df['Sentiment']),  test_size=0.2, shuffle=True, random_state=42)

model1.fit(x=train_x,y=train_y,batch_size=100,epochs=10,verbose=1,shuffle=True,validation_split=0)


# Using Attention model
def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

def call(self, x):
    et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
    at = K.softmax(et)
    at = K.expand_dims(at, axis=-1)
    output = x * at
    return K.sum(output, axis=1)
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

inputs=Input((32,))
x=Embedding(input_dim=32+1,output_dim=32,input_length=32,\
            embeddings_regularizer=keras.regularizers.l2(.001))(inputs)
att_in=LSTM(100,return_sequences=True,dropout=0.3,recurrent_dropout=0.2)(x)
att_out=attention()(att_in)
outputs=Dense(1,activation='sigmoid',trainable=True)(att_out)
model=Model(inputs,outputs)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x=train_x,y=train_y,batch_size=100,epochs=10,verbose=1,shuffle=True,validation_split=0.2)
