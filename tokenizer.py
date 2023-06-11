import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat'
]

tokenizer = Tokenizer(num_words = 100)

"""
if we do not want the tokenizer to omit words during tokenisation of 
sentences in the testing data for words that exist only in testing 
sentences but not in training sentences i.e. out of vocab words, we 
need to define some token value for out of vocab words using a 
parameter "oov_token" as shown below
"""

# tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)