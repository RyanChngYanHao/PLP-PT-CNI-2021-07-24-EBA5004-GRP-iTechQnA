"""
SVM Classifier Model - Python question - 1 / No Python - 0
"""
import shutup; shutup.please()

import pickle
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def clf1(q):
    def pre_process(text):
        tokens = nltk.word_tokenize(text.lower())
        wnl = nltk.WordNetLemmatizer()
        tokens=[wnl.lemmatize(t) for t in tokens]
        tokens=[word for word in tokens if word not in stopwords.words('english')]
        text_after_process = " ".join(tokens)
        return(text_after_process)
    
    modelpath = './saved_data/clf1.sav'
    text_clf = pickle.load(open(modelpath, 'rb'))
    q = pre_process(q)
    predicted = text_clf.predict([q]) 
    return predicted[0]
