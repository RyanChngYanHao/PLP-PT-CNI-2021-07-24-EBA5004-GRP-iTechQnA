"""
### Basic Chat Interface ###

Overall:
    init ->
    
    intro -> 
    request question ->
    
    response 1 -> 
    request number of records ->
    
    response 2 ->    
    exit -> if no go back to (request question)

Response 1:
    return question ->
    output classifer result for clf1 (Is Python?) ->
    output classifer result for clf2 (Suggest Tags) ->
    Break question into ROOTS and dobj ->
    Request for number of records to display before matching ->
    
Response 2:
    match ROOTS and ENTITIES to sample data (from source) 
        by cosine similarity and simple rules ->
    display question_id, title, answer summary from text summary model ->
    option to display original QA ->

"""
import shutup; shutup.please()
import os
import sys
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable all tensorflow warnings

from clf1 import clf1
from clf2 import clf2
from ie3 import ie3
from ts4 import summarize

class QA:
    def __init__(self, cont=['n', 'N', 'exit', 'No', 'no', 'bye', 0]):
        self.cont = cont
        
    def begin(self):      
        
        self.intro()
        
        dn_exit = True
        while dn_exit:
            # Initiate question request            
            print('May I get your Python-related question?') 
            print('[Enter] + Ctrl-Z/D + [Enter] to submit:\n\t\t')
            qlist = sys.stdin.readlines()
            q = ' '.join(qlist)            
            print('\n')          
            # return first response after getting question
            self.response_1(q) 
            # continue if likely python
            if clf1(q):
                # return second response after getting question
                self.response_2(q)
            
            q_exit = input('Any more questions? (Y/N) \n\t\t')
            print('\n')            
            
            if q_exit in self.cont:                
                print('\tThank you for using iTechQnA!')
                print('\tYou have chosen to exit...')
                print('\n')
                dn_exit = False
            else:                
                print('\tLet\'s continue.')
                self.clear_screen()
                
    def clear_screen(self):
        clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')
        clearConsole()
        
    def intro(self):
        # https://www.delftstack.com/howto/python/python-clear-console/        
        self.clear_screen()
        print('<<<Welcome to iTechQnA 2021, developed by NUS-ISS MTech IS AI Apprentices>>>\n')
        print('iTechQnA takes Python-related questions using Python,\n returning best found answers using samples from StackOverflow corpus (2018-2021/06)\n')
        print('iTechQnA has 4 components:\n')
        print('\t 1. Is Python Classifier --- tfidf + SVM model')
        print('\t 2. Tags Classifier --- BiLSTM Encoder / LSTM Decoder Sequence to Sequence model')
        print('\t 3. Basic ROOTS & dObj Extraction --- Rule-based by word token dependency (NLP)')
        print('\t 4. Text Summary --- Using manually built transformer\n')
        print('Available on https://github.com/RyanChngYanHao/PLP-PT-CNI-2021-07-24-EBA5004-GRP-iTechQnA')
        print('\n')
        
        
    def response_1(self, q):
        print('Questions received as:')
        print(f'{q}')                              
        print('\tIs Python:')
        print(f'\t\t>>>\t{self.is_py(q)}')
        if clf1(q):                  
            print('\tPossible tags are:')
            print(f'\t\t>>>\t{self.suggest_tags(q)}')
        else:
            print('\tPossible tags are:')
            print('\t\t>>>\tNon-applicable')
        print('\tRoots detected is/are:')
        print(f'\t\t>>>\t{self.q_ie(q)[0]}')
        print('\tInterested dObj detected is/are:')
        print(f'\t\t>>>\t{self.q_ie(q)[1]}')
        print('\n')               
        return
    
    def response_2(self, q):
        # match
        print('Retrieving matches...') 
        rows = int(input('How many matches are you looking for? (Enter an integer, max is 5, default is 1)\n\t'))
        print('\n')
        try:
            if rows<=5:
                df = self.match(q, rows)
            else: 
                df = self.match(q, 5)
        except:
            df = self.match(q, 1) 
        # view original answers
        view = input('Would you like to view original QA? (Y/N)\n\t')
        print('\n')
        if view not in self.cont:
            for j in range(rows):
                print('_' * 150)
                print(f'\t\t\t<<< Record {j+1} >>>')
                print('_' * 150)
                print('\t\t\tORIGINAL Question (start)\n\n')
                print('\t\t\tStackOverflow question_id:')
                print(f"\t\t\t\t>>>\t{df['q_qid'].iloc[j]}")
                print('\t\t\tQuestion creation date:')
                print(f"\t\t\t\t>>>\t{df['q_creation_date'].iloc[j]}")  
                print('_' * 150)
                clean_q = self.clean_text(df['q_body'].iloc[j])
                print(clean_q, end='\n')
                print('_' * 150)
                print('\t\t\tORIGINAL Question (end)\n\n')
                print('_' * 150)
                input("Press Enter to continue...")
                print('_' * 150)
                print('\t\t\tORIGINAL Answer (start)\n\n')
                print('\t\t\tStackOverflow answer_id:')
                print(f"\t\t\t\t>>>\t{df['a_aid'].iloc[j]}")
                print('_' * 150)
                clean_a = self.clean_text(df['a_body'].iloc[j])
                print(clean_a, end='\n')
                print('_' * 150)
                print('\t\t\tORIGINAL Answer (end)\n\n')
                print('_' * 150)
                input("Press Enter to continue...")
        return
        
    def is_py(self, q):        
        # clf1 predict python - 1 / not python - 0
        if clf1(q):
            reply = 'Highly likely'
        else:
            reply = 'Pretty unlikely'
        return reply

    def suggest_tags(self, q):
        # classifier(q) to return text       
        reply = clf2(q)       
        return reply
    
    def q_ie(self, q):
        # break q to root and dobj
        roots, ents = ie3().get_ie(q)
        return roots, ents
    
    def match(self, q, rows):
        df = ie3().best(q, rows)
        for i in range(rows):
            print(f'Record no. {i+1}')
            # print('\tStackOverflow question_id:')
            # print(f"\t\t>>>\t{df['q_qid'].iloc[i]}")
            print('\tQuestion Title:')
            print(f"\t\t>>>\t{df['q_title'].iloc[i]}")            
            print('\tCombined Similarity Score (naive == root * entity):')
            print(f"\t\t>>>\t{df['combine_score'].iloc[i]}")  
            print('\t\t\tRoots to roots similarity:')
            print(f"\t\t\t\t>>>\t{df['roots_score'].iloc[i]}")
            print('\t\t\tEntities to entities similarity:')
            print(f"\t\t\t\t>>>\t{df['ents_score'].iloc[i]}")
            summary = summarize(df['a_body'].iloc[i])            
            print('\tiTechQnA summary of the accepted answer:')
            print(f"\t\t>>>\t{summary}")
            print('\n')
            input("Press Enter to continue...")
        return df   
    
    def clean_text(self, text):
      reg = re.compile('<.*?>')
      clean = re.sub(reg, '', text)
      return clean
    
if __name__ == '__main__':
    QA().begin()
    

