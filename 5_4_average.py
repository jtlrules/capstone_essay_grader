import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Load data
data = pd.read_csv('data.csv', sep=',', encoding="iso8859_15")

# Tokenize text
tokenizer = Tokenizer(num_words=10000, lower=True)
tokenizer.fit_on_texts(data['essay'].values)
X = tokenizer.texts_to_sequences(data['essay'].values)
X = pad_sequences(X)

# Create embedding matrix using GloVe
embedding_dict = {}
embedding_dim=300
num_words=10000
with open("GloVe/glove.42B.300d.txt", 'r', encoding='iso8859_15') as f:
    for line in f:
        try:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vectors
        except:
            continue
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > num_words - 1:
        break
    else:
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# Average word vectors for each essay
essay_vectors = np.zeros((len(X), embedding_dim))
for i, essay in enumerate(X):
    word_vectors = np.zeros((len(essay), embedding_dim))
    for j, word_index in enumerate(essay):
        if word_index != 0:
            word_vectors[j] = embedding_matrix[word_index]
    essay_vectors[i] = np.mean(word_vectors, axis=0)

# Prepare target variable
y = data['score'].values

# split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(essay_vectors, y, test_size=0.2, random_state=42)

# Create model
model = keras.Sequential()
#model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(embedding_dim,))))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(embedding_dim, activation='relu', input_shape=(embedding_dim,)))
model.add(layers.Dropout(0.5))
#model.add(layers.Dense(128, activation='relu', input_shape=(embedding_dim,)))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(Bidirectional(LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile model
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Evaluate neural network
score = model.evaluate(X_test, y_test, batch_size=16)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#print('Test precision:', score[2])

pred = model.predict(X_test)
difference=[]
for i in range(len(X_test)):
    difference.append(abs(y_test[i]-pred[i]))

prediction_dict={'actual': y_test.tolist(),
                 'predicted': pred.tolist(),
                 'difference: ': difference}

print("Mean difference:", np.mean(difference))
print("Median difference:", np.median(difference))
print("Standard Deviation: ", np.std(difference))
print("Min difference:", min(difference))
print("Max difference:", max(difference))

pred_frame = pd.DataFrame.from_dict(prediction_dict)

pred_frame.to_csv("predictions.csv", encoding="iso8859_15", index=False)