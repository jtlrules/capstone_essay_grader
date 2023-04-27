# Natural Language Toolkit
import nltk

# Stop-words
from nltk.corpus import stopwords
stop_words = set(nltk.corpus.stopwords.words('english'))

# Tokenizing
from nltk import word_tokenize, sent_tokenize

# Keras
import keras
from keras import *


# https://www.kaggle.com/code/rajmehra03/a-detailed-explanation-of-keras-embedding-layer
# Need to find length of maximum doucument - every embedding layer needs to be same length. Pad shorter docs with 0
maxlen = -1
for doc in corp:
    tokens = nltk.word_tokenize(doc)
    if (maxlen < len(tokens)):
        maxlen = len(tokens)

print("The maximum number of words in any document is : ", maxlen)
# Via excel: Max words = 1064

# To create Keras embedding layer:
# 1. sent_tokenize the doc into sentences
# 2. Each sentence has a list of words which we will one_hot encode (one_hot function)
# 3. Each sentence have different # of words. Pad
# 4. 

# Keras doc Embedding layer:
# https://keras.io/api/layers/core_layers/embedding/

# Keras TextVectorization layer:
# https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/#textvectorization-class