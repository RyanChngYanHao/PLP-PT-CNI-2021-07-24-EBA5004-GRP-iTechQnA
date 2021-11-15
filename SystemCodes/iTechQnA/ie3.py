"""
Rules, Roots and Entities
"""
#%% Load
import pandas as pd
import re
import spacy
nlp = spacy.load("en_core_web_sm")
import math
from collections import Counter

import nltk
wnl = nltk.WordNetLemmatizer()

filename1 = './saved_data/qa_ents.csv'
df_qa_ents = pd.read_csv(filename1)
filename2 = './saved_data/ents.csv'
df_ents = pd.read_csv(filename2)

#%% Class ie3
class ie3:
    def __init__(self):
        pass
    
    def get_ie(self, q):
           # break q to root and dobj           
           q = q.lower()           
           doc = nlp(q)           
           root_list = [token.text for token in doc if token.dep_ == 'ROOT']           
           roots = ' '.join(root_list) 
           
           ent_list = [wnl.lemmatize(token.text) for token in doc if token.dep_ == 'dobj']           
           # direct word match to tag
           q = re.sub('[^a-zA-Z \n]', '', q)
           q_tokens = [t for t in q.split()]
           q_match = [t for t in q_tokens if t in df_ents['ents'].tolist()]          
           ent_list = ent_list + q_match          
           ents = ' '.join(ent_list)        
           return roots, ents

    def compute_similarity(self, text1, text2):
        
        # lemmatize to reduce differences
        def lemma(text):            
            lemmatize_text = ' '.join([wnl.lemmatize(t) for t in text.split()])
            return lemmatize_text
        
        # cosine similarity
        # https://gist.github.com/ahmetalsan/06596e3f2ea3182e185a
        def get_cosine(vec1, vec2):
            intersection = set(vec1.keys()) & set(vec2.keys())
            numerator = sum([vec1[x] * vec2[x] for x in intersection])
        
            sum1 = sum([vec1[x]**2 for x in vec1.keys()])
            sum2 = sum([vec2[x]**2 for x in vec2.keys()])
            denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
            if not denominator:
                return 0.0
            else:
                return float(numerator) / denominator
            
        def text_to_vector(text):
            word = re.compile(r'\w+')
            words = word.findall(text)
            return Counter(words)
        
        def get_result(content_a, content_b):
            text1 = content_a
            text2 = content_b
        
            vector1 = text_to_vector(text1)
            vector2 = text_to_vector(text2)
        
            cosine_result = get_cosine(vector1, vector2)
            return cosine_result
        
        l_text1 = lemma(str(text1))
        l_text2 = lemma(str(text2))
        score = get_result(l_text1, l_text2)
        # non zro threshold
        tscore = max(0.01, round(score,2))
        
        return tscore
        
    def best(self, q, rows=1):        
        q_roots, q_ents = self.get_ie(q)
        df_qa_ents['roots_score'] = [self.compute_similarity(q_roots, r) for r in df_qa_ents['roots']]
        df_qa_ents['ents_score'] = [self.compute_similarity(q_ents, e) for e in df_qa_ents['ents']]
        df_qa_ents['combine_score'] = df_qa_ents['roots_score'] * df_qa_ents['ents_score']
        # top
        df = df_qa_ents.sort_values(by=['combine_score', 'q_score', 'q_creation_date'], ascending=False)
        return df.head(rows)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
         