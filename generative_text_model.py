import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam

with open("irish_lyrics_eof.txt", "r") as f:
    lines = f.readlines()

corpus = [line.lower().rstrip('\n') for line in lines]


## Tokenizing the corpus

tokenizer = Tokenizer() 

# Note: No oov_token specified for encoding as we will be training on the entire corpus 
# (as we don't need a validation dataset for generative text models) and hence, nothing 
# will be out of vocab by defnition

tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1 

# print('total words: {}'.format(total_words))

# Note: 1 is added above (to total_words) to count 0 token as valid word which will be 
# generated when we will start padding sub-sentences from the full corpus


## Generating n-gram sequences

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]


## Creating ys as categorical and one hot encoded

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


## RNN-LSTM Modeling

model = Sequential()
model.add(Embedding(total_words, 240, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)

## Generation of New Text

seed_text = "I made a poetry machine"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)




