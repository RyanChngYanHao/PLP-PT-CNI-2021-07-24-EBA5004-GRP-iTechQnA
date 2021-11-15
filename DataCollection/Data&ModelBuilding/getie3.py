"""
Token dependency of question verus corpus
"""
#%% Load
import pandas as pd
import re
import ast
import spacy
nlp = spacy.load("en_core_web_sm")

filename = './sample_data/qa.csv'
df_qa = pd.read_csv(filename)
#%% Get root_list
def clean_title(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z \n]', ' ', text)
    return text
df_qa['q_title'] = df_qa['q_title'].apply(clean_title)

def get_root_list(text):
    doc = nlp(text)
    root_list = [token.text for token in doc if token.dep_ == 'ROOT']
    return root_list
df_qa['root_list'] = df_qa['q_title'].apply(get_root_list)

# convert to text
df_qa['roots'] = [' '.join(rl) for rl in df_qa['root_list']]
#%% Get ent_list
def get_sub_ent_list1(text):
    # text = re.sub('[\-\.]', r'', text)
    tag_list = ast.literal_eval(text)    
    return tag_list   
df_qa['sub_ent_list1'] = df_qa['q_tags'].apply(get_sub_ent_list1)
    
def get_sub_ent_list2(text):
    doc = nlp(text)
    ent_list = [token.text for token in doc if token.dep_ == 'dobj']
    return ent_list
df_qa['sub_ent_list2'] = df_qa['q_title'].apply(get_sub_ent_list2)

tmpl = []
for i in range(df_qa.shape[0]):
    e12 = df_qa['sub_ent_list1'][i] + df_qa['sub_ent_list2'][i]
    tmpl.append(e12)
df_qa['ent_list'] = tmpl

# remove spaces
df_qa['ent_list'] = [[text.strip(' ') for text in el] for el in df_qa['ent_list']] 
df_qa['ent_list'] = [[text for text in el if text != ''] for el in df_qa['ent_list']]

df_qa['ents'] = [' '.join(el) for el in df_qa['ent_list']]
#%% Save file
df_qa.to_csv('./saved_data/qa_ents.csv', index=False)

ents = []
for l in df_qa['sub_ent_list1'].tolist():
    ents += l
ents = list(set(ents))
ents = pd.DataFrame({'ents': ents})
ents.to_csv('./saved_data/ents.csv', index=False)
