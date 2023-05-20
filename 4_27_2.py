### Read in CSV files

import pandas
import tensorflow

data = pandas.read_csv('data.csv', sep=',', encoding='iso8859_15')

# Create a vocabulary
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size = 10000
max_length = 600
docs = data['essay'].tolist()
encoded_docs = [one_hot(d, vocab_size) for d in docs]

# Pad the sequences
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# Define target
import numpy
target = data['score'].values
target = numpy.reshape(target, (target.shape[0], 1))

# Create sparse matrix
sp = tensorflow.sparse.SparseTensor(indices=np.array(padded_docs.nonzero()).T, 
                            values=padded_docs[padded_docs.nonzero()],
                            dense_shape=padded_docs.shape)

# Reorder the sparse matrix
sp_reordered = tensorflow.sparse.reorder(sp)

### Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(essay_data, target, test_size=0.2)


### Define neural network architecture
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=essay_data.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

### Compile the neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### Train the neural network
from sklearn.utils import shuffle

# Shuffle the training data
X_train, y_train = shuffle(X_train, y_train, random_state=0)

# Sort the indices of the sparse matrix
X_train.sort_indices()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

### Evaluate the neural network
score = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
