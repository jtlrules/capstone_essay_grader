### Read in CSV files
import pandas

trainingSet = pandas.read_csv('Raw data/training_set_rel3.tsv', sep='\t', encoding='iso8859_15')
validSet = pandas.read_csv('Raw data/valid_set.tsv', sep='\t', encoding='iso8859_15')
valid_sample_submission = pandas.read_csv('Raw data/valid_sample_submission_5_column.csv', sep=',', encoding='iso8859_15')
pandas.set_option('display.max_columns', None)
### Preprocess

trainingSet = trainingSet[["essay_id", "essay_set", "essay", "domain1_score", "domain2_score"]]
#trainingSet.set_index("essay_id", inplace=True)

# Score preprocessing
# Training Set
for index, row in trainingSet.iterrows():
    set_id:int = row.essay_set
    domain1_score = row['domain1_score']
    if (domain1_score == 0):
        trainingSet.drop([index])
        index-=1
        continue
    if (set_id == 1):
        trainingSet.loc[index, 'score'] = domain1_score / 12
    elif (set_id == 2):
        domain2_score = row['domain2_score']
        trainingSet.loc[index, 'score'] = (domain1_score+domain2_score) / 10
    elif (set_id == 3):
        trainingSet.loc[index, 'score'] = domain1_score / 3
    elif (set_id == 4):
        trainingSet.loc[index, 'score'] = domain1_score / 3
    elif (set_id == 5):
        trainingSet.loc[index, 'score'] = domain1_score / 4
    elif (set_id == 6):
        trainingSet.loc[index, 'score'] = domain1_score / 4
    elif (set_id == 7):
        trainingSet.drop([index])
        index-=1
        continue
        trainingSet.loc[index, 'score'] = domain1_score / 30
    elif (set_id == 8):
        trainingSet.drop([index])
        index-=1
        continue
        trainingSet.loc[index, 'score'] = domain1_score / 60

# Valid Set
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

scored = pandas.DataFrame.from_dict(score)
validSet = pandas.merge(left=validSet, right=scored, left_on="essay_id", right_on="essay_id")

#print(validSet.head(5))
#print(trainingSet.head(5))
# Merge the 2 sets, so they can be evenly split later
Data = pandas.concat([trainingSet, validSet], axis="rows")
Data = Data[["essay_id", "essay_set", "essay", "score"]]
Data = Data[Data['score'] > 0]
#Data = Data[Data['essay_set'] < 7]
Data.reset_index(inplace=True)
#print(validSet.shape)
#print(trainingSet.shape)
#print(Data.shape)
#print(Data.tail(5))

# Essay preprocessing
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')

ps = PorterStemmer()
for i in range(0, len(Data)):
#    #print(Data['essay'][i])
    essay = re.sub('[^a-zA-Z]', ' ', str(Data['essay'][i]))
    essay = essay.lower()
    essay = essay.split()
    essay = [ps.stem(word) for word in essay if not word in stopwords.words('english')]
    essay = ' '.join(essay)
    Data['essay'][i] = essay

Data.to_csv("data.csv", encoding="iso8859_15", index=False)