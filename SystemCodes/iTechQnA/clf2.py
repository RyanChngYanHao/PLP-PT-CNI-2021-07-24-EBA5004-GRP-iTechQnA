"""
Seq2Seq BiLSTM Encoder LSTM Decoder Model
"""
#%% Load
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable all tensorflow warnings

import pickle
import re
import numpy as np

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from keras.models import Model

#%% Load tokenizers
with open('./saved_data/tokenizer_A.pickle', 'rb') as handle:
    tokenizer_A = pickle.load(handle)
with open('./saved_data/tokenizer_B.pickle', 'rb') as handle:
    tokenizer_B = pickle.load(handle)

"""
>>> inherit from getclf2
vocab_size_A: 5001
maxlen_A: 150


vocab_size_B: 34
maxlen_B: 7
"""    
word_index_A = tokenizer_A.word_index
vocab_size_A = 5001
maxlen_A = 150
word_index_B = tokenizer_B.word_index
vocab_size_B = 34
maxlen_B = 6

#%% Reconstruct and load best model weights
hidden_dim = 2^5 
encoder_inputs = Input(shape=(None, vocab_size_A))
encoder = Bidirectional(LSTM(hidden_dim, return_state=True))
encoder_outputs, state_fh, state_fc, state_bh, state_bc = encoder(encoder_inputs)
state_h = Concatenate()([state_fh, state_bh])
state_c = Concatenate()([state_fc, state_bc])
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None, vocab_size_B))
decoder_lstm = LSTM(hidden_dim*2, return_sequences=True, return_state=True) #define lstm
decoder_outputs, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states) #link: [decoder_input,encoder_states] ----> lstm
decoder_dense = Dense(vocab_size_B, activation='softmax') # define last_dense_layer
decoder_outputs = decoder_dense(decoder_outputs) # link:{[decoder_input,encoder_states] ----> lstm} -> last_dense_layer

model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='Seq2Seq_BirectionalLSTM_Encoder_LSTM_Decoder')
model = model.load_weights('./saved_data/checkpoint').expect_partial()

encoder_model = Model(encoder_inputs, encoder_states, name='BidirectionalLSTM_Encoder') #reusing the [encoder_inputs,encoder_states]
decoder_state_input_h = Input(shape=(hidden_dim*2,))
decoder_state_input_c = Input(shape=(hidden_dim*2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)    
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states, name='LSTM_Decoder')

#%% clf2
def clf2(q):
        
    # Redefine same clean text
    def clean_text(text):
      reg = re.compile('<.*?>')
      filepath = re.compile('c:\*.* ')    
      method = re.compile('__.*?__')
      cl_brackets = re.compile('{*}')  
      sq_brackets = re.compile('[*]') 
      http = re.compile('http*?')
      www = re.compile('www*?')      
      clean = re.sub(reg, '', text)
      clean = re.sub('\n', ' ', clean)
      clean = re.sub(filepath, 'filepath', clean)    
      clean = re.sub(method, 'method', clean)
      clean = re.sub(cl_brackets, 'brackets', clean)
      clean = re.sub(sq_brackets, 'brackets', clean)
      clean = re.sub(http, 'website', clean)
      clean = re.sub(www, 'website', clean)
      clean = re.sub('[^a-zA-Z \n]', ' ', clean)
      clean = ' '.join([w for w in clean.split() if w not in stopwords.words('english')]) 
      clean = ' '.join([w for w in clean.split() if len(w)<20])
      clean = ' '.join([w for w in clean.split()[:100]]) 
      return clean
      
    # Redefine same functions
    def decode_sequence(input_seq,
                    num_decoder_tokens,
                    encoder_model,
                    decoder_model,
                    vocab_B,
                    max_decoder_seq_length,
                    maxlen_A,
                    vocab_size_A):

        if len(input_seq)==0:
            return [vocab_B['endoftags']]   
        encoder_input_text = np.zeros((1, maxlen_A, vocab_size_A), dtype='float32')    
        for t, word_id in enumerate(input_seq):
            encoder_input_text[0, t, word_id] = 1.    
        states_value = encoder_model.predict(encoder_input_text)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, vocab_B['startoftags']] = 1.
        stop_condition = False
        decoded_word_index = []
        
        while not stop_condition: 
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)   
            predict_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_word_index.append(predict_token_index)
            if (predict_token_index == vocab_B['endoftags'] or len(decoded_word_index) > max_decoder_seq_length):
                stop_condition = True
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, predict_token_index] = 1.   
            states_value = [h, c]    
        return decoded_word_index

    def indexSeq_to_text_A(list_of_indices):
        reverse_word_map_A = dict(map(reversed, word_index_A.items()))
        words = [reverse_word_map_A.get(letter) for letter in list_of_indices]
        return(words)
    
    def indexSeq_to_text_B(list_of_indices):
        reverse_word_map_B = dict(map(reversed, word_index_B.items()))
        words = [reverse_word_map_B.get(letter) for letter in list_of_indices]
        return(words)
    
    def testSeq2Sq(listOfSeqA):
        listOfSeqB=[]
        token_seqA = tokenizer_A.texts_to_sequences(listOfSeqA)
        
        for a in token_seqA:
            r = decode_sequence(a,vocab_size_B,encoder_model,decoder_model,word_index_B,maxlen_B,maxlen_A,vocab_size_A)
            tokens_b = indexSeq_to_text_B(r)
            sentB = ' '.join(tokens_b)
            listOfSeqB.append(sentB)            
        return listOfSeqB    

    q = clean_text(q)
    predicted = testSeq2Sq([q])
    pred_clean = predicted[0].replace('endoftags', '')      
    return pred_clean