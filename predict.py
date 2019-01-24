# imports
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from pickle import load
import sys

# load model and tokenizer
model = load_model('model.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))

# open test file (provided as cmd arguement)
with open(sys.argv[1], encoding='utf-8') as file:
    df = pd.read_csv(file)

# pre-process data
df['Question 1'].replace(r'\W+', ' ', regex=True, inplace=True)
df['Question 2'].replace(r'\W+', ' ', regex=True, inplace=True)

x_test1 = list(df.values[:, 0].astype('str'))
x_test2 = list(df.values[:, 1].astype('str'))
max_sequence_len = 50

# convert text to int sequence
x_test1 = tokenizer.texts_to_sequences(x_test1)
x_test2 = tokenizer.texts_to_sequences(x_test2)

# pad sequences
x_test1 = pad_sequences(x_test1, maxlen=max_sequence_len,
                        padding='pre')
x_test2 = pad_sequences(x_test2, maxlen=max_sequence_len,
                        padding='pre')

# predict the results
predictions = model.predict([x_test1, x_test2])

# write out results to output.txt
with open('output.txt', 'w') as out_file:
    for pred in predictions:
        out_file.write(str(pred[0]) + '\n')
