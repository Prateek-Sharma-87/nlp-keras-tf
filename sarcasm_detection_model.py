import json

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importing data

with open("Sarcasm_Headlines_Dataset_Modified.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Slicing data into training and testing

training_size = int(round(0.8*len(sentences), 0))

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Tokenizing and Padding

vocab_size = 30000  # decided after checking the word_index post tokenization of all sentences 
max_length = 40   # decided after tokenization and padding of all sentences without maxlen param
padding_type = 'post'
trunc_type = 'post'

# # To figure out 'vocab_size' and 'max_length' variables
# tokenizer = Tokenizer(oov_token="<OOV>")
# tokenizer.fit_on_texts(sentences)
# word_index = tokenizer.word_index
# print(len(word_index.keys()))

# sequences = tokenizer.texts_to_sequences(sentences)
# sequences_padded = pad_sequences(sequences, padding=padding_type)
# print(sequences_padded.shape)

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=trunc_type)

# Conversion to Array before feeing into the ANN Model

training_padded = np.array(training_padded)
training_labels = np.array(training_labels).reshape([-1, 1])
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels).reshape([-1, 1])

# ANN Modeling

embedding_dim = 20

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), 
    tf.keras.layers.GlobalAveragePooling1D(), 
    tf.keras.layers.Dense(24, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30

model.fit(training_padded, training_labels, epochs=num_epochs, 
          validation_data=(testing_padded, testing_labels), verbose=2)

# Sarcasm detection in New Sentences

sentence = [
    "granny starting to fear spiders in the garden might be real", 
    "the weather today is bright and sunny"
]

sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, 
                       padding=padding_type, truncating=trunc_type)

print(model.predict(padded))
