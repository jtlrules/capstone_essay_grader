### Read in CSV files

import pandas
import tensorflow as tf
import numpy as np

data = pandas.read_csv('data.csv', sep=',', encoding='iso8859_15')

# Convert essay column to numerical representations
#vectorizer = CountVectorizer()
#essay_data = vectorizer.fit_transform(data['essay'])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2)

# Tokenize the text
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data['essay'].tolist())
train_sequences = tokenizer.texts_to_sequences(train_data['essay'].tolist())
test_sequences = tokenizer.texts_to_sequences(test_data['essay'].tolist())

# Pad the sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_length = 550
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Load the pre-trained GloVe word embeddings
embeddings_index = {}
embedding_dim = 200
with open('GloVe/glove.twitter.27B.200d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create an embedding matrix
word_index = tokenizer.word_index
num_words = min(10000, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Convert the target variable to a 2D numpy array
train_target = train_data['score'].values
test_target = test_data['score'].values
train_target = np.reshape(train_target, (train_target.shape[0], 1))
test_target = np.reshape(test_target, (test_target.shape[0], 1))

#target = np.reshape(target, (target.shape[0], 1))

# Create sparse matrix
#train_sp = tf.sparse.SparseTensor(indices=np.array(train_padded_docs.nonzero()).T, 
#                                  values=train_padded_docs[train_padded_docs.nonzero()],
#                                  dense_shape=train_padded_docs.shape)
#test_sp = tf.sparse.SparseTensor(indices=np.array(test_padded_docs.nonzero()).T, 
#                                 values=test_padded_docs[test_padded_docs.nonzero()],
#                                 dense_shape=test_padded_docs.shape)

# Reorder the sparse matrix
#train_sp_reordered = tf.sparse.reorder(train_sp)
#test_sp_reordered = tf.sparse.reorder(test_sp)


### Define neural network architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Define neural network architecture
model = Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(tf.keras.layers.Flatten())
#model.add(Dropout(0.3))
#model.add(LSTM(100))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))
#model.add(LSTM(100))
#model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

### Compile the neural network
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mse', 'accuracy', 'poisson', 'KLDivergence'])

# Shuffle the training data
from sklearn.utils import shuffle
#X_train, y_train = shuffle(X_train, y_train, random_state=0)

# Sort the indices of the sparse matrix
#X_train.sort_indices()

# Train the model
model.fit(train_padded, train_target, epochs=10, batch_size=32)

### Evaluate the neural network
# Evaluate neural network
score = model.evaluate(test_padded, test_target, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


pred = model.predict(test_padded)
difference=[]
for i in range(len(test_padded)):
    difference.append(abs(test_target[i]-pred[i]))

prediction_dict={'actual': test_target.tolist(),
                 'predicted': pred.tolist(),
                 'difference: ': difference}

print("Mean difference:", np.mean(difference))
print("Standard Deviation: ", np.std(difference))
print("Min difference:", min(difference))
print("Max difference:", max(difference))