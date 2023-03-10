# Keras, Tensorflow
# Gensim - Words as semantic vectors - pip install --upgrade gensim
from gensim import corpora, models, similarities, downloader, word2vec

# One-hot representation
#Taken from https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
from numpy import argmax
# define input string
data = 'hello world'
print(data)
# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)
# one hot encode
onehot_encoded = list()
for value in integer_encoded:
 letter = [0 for _ in range(len(alphabet))]
 letter[value] = 1
 onehot_encoded.append(letter)
print(onehot_encoded)
# invert encoding
# argmax finds index of max value in vector
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)
