"""
SVM Classifier Model - Python question - 1 / No Python - 0
With reference to MTech IS 2020-2021 / EBA5004 - TA Workshop

"""
import pandas as pd
import numpy as np
import pickle

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn import metrics 
# from sklearn.linear_model import LogisticRegression

filename = './sample_data/qo.csv'
df = pd.read_csv(filename)
# not required to remove html tags --> tfidf will take care of them later anyway
df['text'] = df['title'].astype(str) + ' ' + df['body'].astype(str)

def pre_process(text):
    tokens = nltk.word_tokenize(text.lower())
    wnl = nltk.WordNetLemmatizer()
    tokens=[wnl.lemmatize(t) for t in tokens]
    tokens=[word for word in tokens if word not in stopwords.words('english')]
    text_after_process = " ".join(tokens)
    return(text_after_process)

df['text'] = df['text'].apply(pre_process)

X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, test_size=0.20, random_state=21)
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', svm.LinearSVC(C=1.0))
                     #('clf', LogisticRegression())
                    ])
text_clf.fit(X_train, y_train)    
predicted = text_clf.predict(X_test) 
print(metrics.confusion_matrix(y_test, predicted))
print(np.mean(predicted == y_test))
print(metrics.classification_report(y_test, predicted))

modelpath = './saved_data/clf1.sav'
pickle.dump(text_clf, open(modelpath, 'wb'))
# text_clf = pickle.load(open(modelpath, 'rb'))

"""
## Logistic Regression
[[5493  499]
 [ 852 5063]]
0.8865373309817755
              precision    recall  f1-score   support

           0       0.87      0.92      0.89      5992
           1       0.91      0.86      0.88      5915

    accuracy                           0.89     11907
   macro avg       0.89      0.89      0.89     11907
weighted avg       0.89      0.89      0.89     11907

## SVM
[[5437  555]
 [ 767 5148]]
0.8889728730998572
              precision    recall  f1-score   support

           0       0.88      0.91      0.89      5992
           1       0.90      0.87      0.89      5915

    accuracy                           0.89     11907
   macro avg       0.89      0.89      0.89     11907
weighted avg       0.89      0.89      0.89     11907
"""
new_question = ['How to concat two columns in pd dataframe?']
predicted = text_clf.predict(new_question) ; print(predicted[0]) # return 1

new_question = ['MainActivity which has a button when on pressed reads data from a QRscan. After that, a background service is started and data from QRscan is sent to it. It has an update method defined to create a fragment once it is invoked by ']
predicted = text_clf.predict(new_question) ; print(predicted[0]) # return 0

