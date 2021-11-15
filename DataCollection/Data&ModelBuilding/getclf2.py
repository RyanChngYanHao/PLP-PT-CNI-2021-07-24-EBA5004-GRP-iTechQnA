"""
Seq2Seq BiLSTM Encoder LSTM Decoder Model
With reference to MTech IS 2020-2021 / EBA5004 - TPML Workshop

"""
#%% Load
import pickle
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import math
from collections import Counter

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from keras.models import Model

filename = './sample_data/qa.csv'
df = pd.read_csv(filename)
#%% Pre-process
def clean_text(text):
  reg = re.compile('<.*?>')  
  clean = re.sub(reg, '', text)
  clean = re.sub('\n', ' ', clean)
  clean = re.sub('[^a-zA-Z \n]', ' ', clean)
# remove stop words
  clean = ' '.join([w for w in clean.split() if w not in stopwords.words('english')]) 
# only keep word of length
  clean = ' '.join([w for w in clean.split() if len(w)<20])
# only keep first n words
  clean = ' '.join([w for w in clean.split()[:150]]) 
  return clean

def clean_tags(tags):  
  clean = tags.replace('python,', '')
  clean = re.sub('[\[\]\-\.\']', '', clean)
  clean = 'startoftags, ' + clean + ', endoftags'
  return clean

df['seq_A'] = df['q_title'].astype(str) + ' ' + df['q_body'].astype(str)
df['seq_A'] = df['seq_A'].apply(clean_text)
df['seq_B'] = df['q_tags'].apply(clean_tags)

# split train test, 'include' as a list of boolean of len(df)
include = np.random.rand(len(df)) <= 0.90
train_text_set = df[include]; eval_set = df[~include]
print(len(train_text_set), len(eval_set))
#%% tokenize
cap_words = 5000
tokenizer_A = Tokenizer(num_words = cap_words)
tokenizer_A.fit_on_texts(train_text_set['seq_A'])
train_text_set['seq_A_vec'] = tokenizer_A.texts_to_sequences(train_text_set['seq_A'])
word_index_A = tokenizer_A.word_index
vocab_size_A = min(len(word_index_A), cap_words) + 1 # Adding 1 because of reserved 0 index by Tokenizer
maxlen_A = max([len(seq) for seq in train_text_set['seq_A_vec']]) # longest text in train set

tokenizer_B = Tokenizer()
tokenizer_B.fit_on_texts(train_text_set['seq_B'])
train_text_set['seq_B_vec'] = tokenizer_B.texts_to_sequences(train_text_set['seq_B'])
word_index_B = tokenizer_B.word_index
vocab_size_B = len(word_index_B) + 1  # Adding 1 because of reserved 0 index by Tokenizer
maxlen_B = max([len(seq) for seq in train_text_set['seq_B_vec']]) # longest text in train set

print('vocab_size_A:', vocab_size_A)
print('maxlen_A:', maxlen_A)
print('\n')
print('vocab_size_B:', vocab_size_B)
print('maxlen_B:', maxlen_B)
print('\n')

with open('./saved_data/tokenizer_A.pickle', 'wb') as handle:
    pickle.dump(tokenizer_A, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./saved_data/tokenizer_B.pickle', 'wb') as handle:
    pickle.dump(tokenizer_B, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
"""
with open('./saved_data/tokenizer_A.pickle', 'rb') as handle:
    tokenizer_A = pickle.load(handle)
with open('./saved_data/tokenizer_B.pickle', 'rb') as handle:
    tokenizer_B = pickle.load(handle)
"""
#%% Encode Decoder
## setup zeros
encoder_input_data = np.zeros((len(train_text_set['seq_A_vec']), maxlen_A, vocab_size_A), dtype='float32')
decoder_input_data = np.zeros((len(train_text_set['seq_A_vec']), maxlen_B, vocab_size_B), dtype='float32')
decoder_target_data = np.zeros((len(train_text_set['seq_A_vec']), maxlen_B, vocab_size_B), dtype='float32')
print(encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)

## populate into 3d arrray in the format (entries, terms, word index)
for i, (input_text, target_text) in enumerate(zip(train_text_set['seq_A_vec'], train_text_set['seq_B_vec'])):
    for t, word_id in enumerate(input_text):
        encoder_input_data[i, t, word_id] = 1.
##    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, word_id in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, word_id] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, word_id] = 1.0
    # decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    # decoder_target_data[i, t:, target_token_index[' ']] = 1.

#%% Build Seq2Seq model using BiLSTM Encoder and LSTM Decoder
hidden_dim = 2^5

# Encoder Part: {encoder_input -> lstm} -> encoder_states
encoder_inputs = Input(shape=(None, vocab_size_A))
encoder = Bidirectional(LSTM(hidden_dim, return_state=True))
encoder_outputs, state_fh, state_fc, state_bh, state_bc = encoder(encoder_inputs)
    # https://stackoverflow.com/questions/50815354/seq2seq-bidirectional-encoder-decoder-in-keras
    # https://www.sciencedirect.com/science/article/pii/S2667305321000387#fig0002
state_h = Concatenate()([state_fh, state_bh])
state_c = Concatenate()([state_fc, state_bc])
encoder_states = [state_h, state_c]

# Decoder Part:{[decoder_input , encoder_states] ----> lstm} --> last_dense_layer
decoder_inputs = Input(shape=(None, vocab_size_B))
decoder_lstm = LSTM(hidden_dim*2, return_sequences=True, return_state=True) #define lstm
decoder_outputs, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states) #link: [decoder_input,encoder_states] ----> lstm
decoder_dense = Dense(vocab_size_B, activation='softmax') # define last_dense_layer
decoder_outputs = decoder_dense(decoder_outputs) # link:{[decoder_input,encoder_states] ----> lstm} -> last_dense_layer

#Link encoder -> decoder 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='Seq2Seq_BirectionalLSTM_Encoder_LSTM_Decoder')
opt = keras.optimizers.RMSprop(learning_rate=0.08)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# set call backs
checkpoint_filepath = './saved_data/checkpoint'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
## if load >> model.load_weights(checkpoint_filepath)

#%% Train the model
hist = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size = 200,
                  epochs = 15,
                  validation_split = 0.2,
                  callbacks=[model_checkpoint_callback]).history

# Save model
model.save('./saved_data/Seq2Seq.h5')

plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('./saved_data/clf2_model_acc.png')

# summarize history for loss
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('./saved_data/clf2_model_loss.png')

with open('./saved_data/traininghistory', 'wb') as handle: 
    pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
with open('./saved_data/traininghistory', 'rb') as handle:
    traininghistory = pickle.load(handle)
"""
#%% Seperately define the models
encoder_model = Model(encoder_inputs, encoder_states, name='BidirectionalLSTM_Encoder') #reusing the [encoder_inputs,encoder_states]
encoder_model.summary()

#define decoder_model seperatly as training stages
#[_h, _c] for decoder LSTM
decoder_state_input_h = Input(shape=(hidden_dim*2,))
decoder_state_input_c = Input(shape=(hidden_dim*2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# [decoder_input, h_t0, c_t0] for decoder LSTM
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
# predict [h_t1, c_t1]
decoder_states = [state_h, state_c]
# predict [target_Seq_t1]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states, name='LSTM_Decoder')
decoder_model.summary()

#%% Decode & Reversal
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

    # Encode the input as state vectors.
    ##Encoder(kopi o <pad> <pad> <pad>) = (h0,c0)

    encoder_input_text = np.zeros((1, maxlen_A, vocab_size_A), dtype='float32')    
    for t, word_id in enumerate(input_seq):
        encoder_input_text[0, t, word_id] = 1.

    states_value = encoder_model.predict(encoder_input_text)

    # Generate empty target sequence of word_token.
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Initialize with $start
    target_seq[0, 0, vocab_B['startoftags']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1 (greedy decoding)).
    stop_condition = False
    decoded_word_index = []
    
    while not stop_condition:
        ##(h0,c0) +  $START ----decoder-----> (???,  (h1,c1)) 
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)      

        # Predict the best token (black)
        predict_token_index = np.argmax(output_tokens[0, -1, :])
        decoded_word_index.append(predict_token_index)

        # Exit condition: either hit max length # or find stop character.
        if (predict_token_index == vocab_B['endoftags'] or len(decoded_word_index) > max_decoder_seq_length):
            stop_condition = True

        ##(h1,c1)	+  black -----decoder---->  (???, (h2,c2))

        # Update the target sequence to the predict word_token
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, predict_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_word_index
# Creating a reverse dictionary
# Function takes a tokenized sentence and returns the words
def indexSeq_to_text_A(list_of_indices):
    reverse_word_map_A = dict(map(reversed, tokenizer_A.word_index.items()))
    # Looking up words in dictionary
    words = [reverse_word_map_A.get(letter) for letter in list_of_indices]
    return(words)

def indexSeq_to_text_B(list_of_indices):
    reverse_word_map_B = dict(map(reversed, tokenizer_B.word_index.items()))
    # Looking up words in dictionary
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
#%% Evaluation on unseen set
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

eval_set['predicted'] = testSeq2Sq(eval_set['seq_A'])
eval_set

# remove noise
eval_set['actual'] = eval_set['seq_B'].str.replace('startoftags, ', '').str.replace(', endoftags', '').str.replace(',', '')
eval_set['pred_clean'] = eval_set['predicted'].str.replace(' endoftags', '')
eval_set.head()

eval_set['cos_sim_score'] = eval_set.apply(lambda x: round(get_result(x.pred_clean, x.actual), 2), axis=1)
acc = round(eval_set['cos_sim_score'].sum()/len(eval_set['cos_sim_score']),2)
print(f'Accuracy: {acc}')
print(eval_set['cos_sim_score'].value_counts())
eval_set.to_csv('./saved_data/clf2_eval.csv', index=False)

"""
vocab_size_A: 5001
maxlen_A: 150


vocab_size_B: 34
maxlen_B: 7


(6877, 150, 5001) (6877, 7, 34) (6877, 7, 34)
Model: "Seq2Seq_BirectionalLSTM_Encoder_LSTM_Decoder"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, None, 5001)] 0                                            
__________________________________________________________________________________________________
bidirectional (Bidirectional)   [(None, 14), (None,  280504      input_1[0][0]                    
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, None, 34)]   0                                            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 14)           0           bidirectional[0][1]              
                                                                 bidirectional[0][3]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 14)           0           bidirectional[0][2]              
                                                                 bidirectional[0][4]              
__________________________________________________________________________________________________
lstm_1 (LSTM)                   [(None, None, 14), ( 2744        input_2[0][0]                    
                                                                 concatenate[0][0]                
                                                                 concatenate_1[0][0]              
__________________________________________________________________________________________________
dense (Dense)                   (None, None, 34)     510         lstm_1[0][0]                     
==================================================================================================
Total params: 283,758
Trainable params: 283,758
Non-trainable params: 0

Epoch 1/15
28/28 [==============================] - ETA: 0s - loss: 0.8604 - accuracy: 0.3812
28/28 [==============================] - 59s 2s/step - loss: 0.8604 - accuracy: 0.3812 - val_loss: 0.5719 - val_accuracy: 0.4526
Epoch 2/15
28/28 [==============================] - 43s 2s/step - loss: 0.5278 - accuracy: 0.4559 - val_loss: 0.4718 - val_accuracy: 0.4716
Epoch 3/15
28/28 [==============================] - 47s 2s/step - loss: 0.4425 - accuracy: 0.4742 - val_loss: 0.4452 - val_accuracy: 0.4778
Epoch 4/15
28/28 [==============================] - 44s 2s/step - loss: 0.3994 - accuracy: 0.4832 - val_loss: 0.5002 - val_accuracy: 0.4634
Epoch 5/15
28/28 [==============================] - 46s 2s/step - loss: 0.4291 - accuracy: 0.4852 - val_loss: 0.4264 - val_accuracy: 0.4876
Epoch 6/15
28/28 [==============================] - 52s 2s/step - loss: 0.3980 - accuracy: 0.4896 - val_loss: 0.4102 - val_accuracy: 0.4908
Epoch 7/15
28/28 [==============================] - 41s 1s/step - loss: 0.3724 - accuracy: 0.4946 - val_loss: 0.3809 - val_accuracy: 0.4898
Epoch 8/15
28/28 [==============================] - 54s 2s/step - loss: 0.3401 - accuracy: 0.4990 - val_loss: 0.3738 - val_accuracy: 0.4916 * Best
Epoch 9/15
28/28 [==============================] - 40s 1s/step - loss: 0.3226 - accuracy: 0.5009 - val_loss: 0.3676 - val_accuracy: 0.4908
Epoch 10/15
28/28 [==============================] - 51s 2s/step - loss: 0.3086 - accuracy: 0.5050 - val_loss: 0.3665 - val_accuracy: 0.4876
Epoch 11/15
28/28 [==============================] - 44s 2s/step - loss: 0.2986 - accuracy: 0.5083 - val_loss: 0.3672 - val_accuracy: 0.4901
Epoch 12/15
28/28 [==============================] - 56s 2s/step - loss: 0.2841 - accuracy: 0.5132 - val_loss: 0.3756 - val_accuracy: 0.4819
Epoch 13/15
28/28 [==============================] - 41s 1s/step - loss: 0.2753 - accuracy: 0.5161 - val_loss: 0.3655 - val_accuracy: 0.4886
Epoch 14/15
28/28 [==============================] - 49s 2s/step - loss: 0.2644 - accuracy: 0.5202 - val_loss: 0.3732 - val_accuracy: 0.4834
Epoch 15/15
28/28 [==============================] - 35s 1s/step - loss: 0.2541 - accuracy: 0.5253 - val_loss: 0.3639 - val_accuracy: 0.4844   

Accuracy: 0.78
1.00    304
0.67    282
0.33     79
0.87     54
0.58     12
0.77      3
0.26      2
Name: cos_sim_score, dtype: int64

"""










