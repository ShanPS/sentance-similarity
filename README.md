# sentance-similarity
This is to perform sentence similarity checking using siamese LSTM model. In this model we will pass both the sentence through same LSTM layer to get the encoding for given sequence. Then we will concatenate the encodings and feed it to feed forward network. 


* Run 'model.py' once (with training data in same location) to get model.h5 and tokenizer.pkl files (which are used by 'predict.py').
Then run 'predict.py' by providing path to test file as arguement.
