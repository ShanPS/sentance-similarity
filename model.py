# imports
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from pickle import dump

# setting seed for result consistency
np.random.seed(3)

epochs = 10
batch_size = 128

# read data
df = pd.read_csv('dataset.csv')

# pre-process the data
df['Question 1'].replace(r'\W+', ' ', regex=True, inplace=True)
df['Question 2'].replace(r'\W+', ' ', regex=True, inplace=True)

x_train1 = df.values[:, 0].astype('str')
x_train2 = df.values[:, 1].astype('str')
y_train = df.values[:, 2].astype('int32')

x_train1 = list(x_train1)
x_train2 = list(x_train2)
y_train = y_train.reshape(-1, 1)

# tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train1 + x_train2)
vocab_len = len(tokenizer.word_index) + 1

# convert text to int sequence
x_train1 = tokenizer.texts_to_sequences(x_train1)
x_train2 = tokenizer.texts_to_sequences(x_train2)

# max_sequence_len = 50 , as it gives a good measure
max_sequence_len = 50

# pad the sequences
x_train1 = pad_sequences(x_train1, maxlen=max_sequence_len,
                         padding='pre')
x_train2 = pad_sequences(x_train2, maxlen=max_sequence_len,
                         padding='pre')

# model - siamese lstm
inp1 = Input(shape=(max_sequence_len, ), name='sentence_1')
inp2 = Input(shape=(max_sequence_len, ), name='sentence_2')
emb = Embedding(output_dim=40, input_dim=vocab_len,
                input_length=max_sequence_len)
encoder = LSTM(80)
e1 = encoder(emb(inp1))
e2 = encoder(emb(inp2))
x = concatenate([e1, e2])
x = Dense(20, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inp1, inp2], outputs=out)
model.summary()

# compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train using training data
model.fit([x_train1, x_train2], y_train, epochs=epochs,
          batch_size=batch_size, verbose=2, validation_split=0.2)

# save model and tokenizer details for later to be used by predict.py
model.save('model.h5')
dump(tokenizer, open('tokenizer.pkl', 'wb'))
