from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

print(word_index)

#### padding pre sentences ####
padded = pad_sequences(sequences)

print(sequences)
print(padded)

# Output ->

# {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
# [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
# [[ 0  0  0  5  3  2  4]
#  [ 0  0  0  5  3  2  7]
#  [ 0  0  0  6  3  2  4]
#  [ 8  6  9  2  4 10 11]]

#### padding post sentences ####
padded = pad_sequences(sequences, padding='post')

print(sequences)
print(padded)

# Output ->

# {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
# [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
[[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
# [[ 5  3  2  4  0  0  0]
#  [ 5  3  2  7  0  0  0]
#  [ 6  3  2  4  0  0  0]
#  [ 8  6  9  2  4 10 11]]

