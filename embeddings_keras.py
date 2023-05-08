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

# Pandas - CSV / Data Frame
import pandas

# Read TSV into data frame
# essay_id	essay_set	essay
# rater1_domain1	rater2_domain1	rater3_domain1	domain1_score	rater1_domain2	rater2_domain2	domain2_score	rater1_trait1	rater1_trait2	rater1_trait3	rater1_trait4	rater1_trait5	rater1_trait6	rater2_trait1	rater2_trait2	rater2_trait3	rater2_trait4	rater2_trait5	rater2_trait6	rater3_trait1	rater3_trait2	rater3_trait3	rater3_trait4	rater3_trait5	rater3_trait6
csvData = pandas.read_csv("Raw Data/training_set_rel3.tsv", delimiter="\t")
textOnly = csvData["essay"]

# Preprocessing
essay_features = csvData.copy()
    # Remove graded columns
essay_labels = essay_features.pop('survived')

# Put into Keras TextVectorization Layer

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
text_vectorizer = keras.layers.TextVectorization(
                    standardize="lower_and_strip_punctuation",
                    output_mode="int",
                    )

                    