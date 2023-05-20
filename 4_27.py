### Read in CSV files

import pandas as pd

trainingSet = pd.read_csv('Raw data/training_set_rel3.tsv', sep='\t', encoding='iso8859_15')
validSet = pd.read_csv('Raw data/valid_set.tsv', sep='\t', encoding='iso8859_15')
valid_sample_submission = pd.read_csv('Raw data/valid_sample_submission_5_column.csv', sep=',', encoding='iso8859_15')
### Preprocess

trainingSet = trainingSet[["essay_id", "essay_set", "essay", "domain1_score", "domain2_score"]]
trainingSet.set_index("essay_id", inplace=True)

for idx, row in trainingSet.iterrows():


score = {'essay_id':[],
         'score':[]}
for idx, row in validSet.iterrows():
    set_id:int = row.essay_set
    essay_id:int = row.essay_id

    domain1_pred_id = row['domain1_predictionid']
    
    domain1_row = valid_sample_submission[valid_sample_submission['prediction_id'] == domain1_pred_id]
    domain1_score = domain1_row['predicted_score'].values[0]
    row_score = domain1_score
    
    if (set_id == 1):
        row_score /= 12
    elif (set_id == 2):
        domain2_pred_id = row['domain2_predictionid']
        domain2_row = valid_sample_submission[valid_sample_submission['prediction_id'] == domain2_pred_id]
        domain2_score = domain2_row['predicted_score'].values[0]
        row_score += domain2_score
        row_score /= 10
    elif (set_id == 3):
        row_score /= 3
    elif (set_id == 4):
        row_score /= 3
    elif (set_id == 5):
        row_score /= 4
    elif (set_id == 6):
        row_score /= 4
    elif (set_id == 7):
        row_score /= 30
    elif (set_id == 8):
        row_score /= 60

    score['essay_id'].append(essay_id)
    score['score'].append(row_score)


from sklearn.feature_extraction.text import CountVectorizer

# Convert essay column to numerical representations
vectorizer = CountVectorizer()
essay_data = vectorizer.fit_transform(data['essay'])

### Split data into training and trsting sets
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(essay_data, target, test_size=0.2)

### Define neural network architecture
from keras.models import Sequential
from keras.layers import Dense

# Define neural network architecture
model = Sequential()
model.add(Dense(64, input_dim=essay_data.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))

### Compile the neural network
# Compile neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### Train the neural network
# Train neural network
model.fit(X_train, y_train, epochs=10, batch_size=32)

### Evaluate the neural network
# Evaluate neural network
score = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
