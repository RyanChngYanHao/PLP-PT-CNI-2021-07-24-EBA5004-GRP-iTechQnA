"""
Text Summary - q_body as document, q_title as summary
With reference to MTech IS 2020-2021 / EBA5004 - TPML Workshop
"""
#%% Load
import pandas as pd
import tensorflow as tf
import numpy as np
import re
import datetime
import time
import pickle

filename = './sample_data/qa.csv'
df = pd.read_csv(filename)

#%% tokenize
filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

def clean_body(text):
    codings = re.compile('<code>.*?</code>')
    reg = re.compile('<.*?>') 
    clean = re.sub(codings, '', text)
    clean = re.sub(reg, '', text)
    return clean

df['document'] = df['q_body'].apply(clean_body)
document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
document_tokenizer.fit_on_texts(df['document'])
inputs = document_tokenizer.texts_to_sequences(df['document'])

df['summary'] = df['q_title'].apply(lambda x: '<go> ' + x + ' <stop>')
summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
summary_tokenizer.fit_on_texts(df['summary'])
targets = summary_tokenizer.texts_to_sequences(df['summary'])

#%% Encoder Decoder
encoder_vocab_size = len(document_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1

document_lengths = pd.Series([len(x) for x in df['document']])
summary_lengths = pd.Series([len(x) for x in df['summary']])

print(encoder_vocab_size, decoder_vocab_size)
print(df.shape[0])
print(document_lengths.describe())
print(summary_lengths.describe())

"""
59276 4259
7613
count     7613.000000
mean      1291.564823
std       1453.953798
min         85.000000
25%        602.000000
50%        929.000000
75%       1464.000000
max      29298.000000
dtype: float64
count    7613.000000
mean       73.067910
std        21.478826
min        27.000000
25%        58.000000
50%        70.000000
75%        85.000000
max       175.000000
dtype: float64
"""
encoder_maxlen = 250 # min (Memory constraint)
decoder_maxlen = 90 # ~75%
#%% Truncating and Padding
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')
#%% Setup
# set tf type
inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)
# set buffer and batch
BUFFER_SIZE = 6000  # should be set at least equal to the size of the dataset
BATCH_SIZE = 64
# dataset
dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# Position encoding
def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
# Masking
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
# Scaled Dot Product
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9) 

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
# Multi-headed attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
            
        return output, attention_weights
# Forward-feed Network
def point_wise_feed_forward_network(d_model, dff): # dff is no of neurons in the layer
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
# Fundamental Unit of Transformer encoder
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
# Fundamental Unit of Transformer decoder
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2
# Encoder consisting of multiple EncoderLayer(s)
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
    
        return x
# Decoder consisting of multiple DecoderLayer(s)   
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
        return x, attention_weights
#%% Transformer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
#%% Training
# hyper-params
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
EPOCHS = 10

# Adam optimizer with custom learning rate scheduling
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Defining losses and other metrics
learning_rate = CustomSchedule(d_model)
print (learning_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
train_loss = tf.keras.metrics.Mean(name='train_loss')

# Transformer
transformer = Transformer(
    num_layers, 
    d_model, 
    num_heads, 
    dff,
    encoder_vocab_size, 
    decoder_vocab_size, 
    pe_input=encoder_vocab_size, 
    pe_target=decoder_vocab_size,
)
# Mask
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return enc_padding_mask, combined_mask, dec_padding_mask
# Check Points
checkpoint_path = "./saved_data/ts_checkpoints"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
# Training Steps
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp, 
            True, 
            enc_padding_mask, 
            combined_mask, 
            dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
#%% Start Training
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
  
    for (batch, (inp, tar)) in enumerate(dataset):
        train_step(inp, tar)
    
        # 55k samples
        # we display 3 batch results -- 0th, middle and last one (approx)
        # 7.6 / 64 ~ 120; 120 / 2 = 60
        if batch % 60 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, train_loss.result()))
      
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
    
    print ('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
"""
Epoch 1 Batch 0 Loss 8.3903
Epoch 1 Batch 60 Loss 8.2736
Epoch 1 Loss 8.1028
Time taken for 1 epoch: 707.1963200569153 secs

Epoch 2 Batch 0 Loss 7.7571
Epoch 2 Batch 60 Loss 7.5669
Epoch 2 Loss 7.3354
Time taken for 1 epoch: 759.3896052837372 secs

Epoch 3 Batch 0 Loss 6.8497
Epoch 3 Batch 60 Loss 6.5344
Epoch 3 Loss 6.2971
Time taken for 1 epoch: 796.3612921237946 secs

Epoch 4 Batch 0 Loss 5.7612
Epoch 4 Batch 60 Loss 5.6903
Epoch 4 Loss 5.6021
Time taken for 1 epoch: 796.8010921478271 secs

Epoch 5 Batch 0 Loss 5.4169
Epoch 5 Batch 60 Loss 5.2963
Saving checkpoint for epoch 5 at ./saved_data/ts_checkpoints\ckpt-1
Epoch 5 Loss 5.2296
Time taken for 1 epoch: 797.7722651958466 secs

Epoch 6 Batch 0 Loss 4.9472
Epoch 6 Batch 60 Loss 4.9675
Epoch 6 Loss 4.9229
Time taken for 1 epoch: 805.3773310184479 secs

Epoch 7 Batch 0 Loss 4.6518
Epoch 7 Batch 60 Loss 4.7214
Epoch 7 Loss 4.6853
Time taken for 1 epoch: 795.7363505363464 secs

Epoch 8 Batch 0 Loss 4.4780
Epoch 8 Batch 60 Loss 4.5211
Epoch 8 Loss 4.5057
Time taken for 1 epoch: 795.3869533538818 secs

Epoch 9 Batch 0 Loss 4.3307
Epoch 9 Batch 60 Loss 4.3513
Epoch 9 Loss 4.3417
Time taken for 1 epoch: 737.445506811142 secs



"""
#%% Inference
def evaluate(input_document):
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [summary_tokenizer.word_index["<go>"]]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == summary_tokenizer.word_index["<stop>"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def summarize(input_document):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    summarized = evaluate(input_document=input_document)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    return summary_tokenizer.sequences_to_texts(summarized)[0]  # since there is just one translated document

b = datetime.datetime.now()

# try on some answers (all answers are unseen as training was done q againat q title)
unseen_1 = clean_body(df['a_body'].iloc[1])
ans1 = summarize(unseen_1)
print(ans1)
print(unseen_1)

"""
how to get the elements of a list of a list of a list of a list
########################################################
You just need to recompute a new random number each time.

for x in range(0, 3):
    sent = random.randrange(0,1100)
    print(Words[sent])


Though what might be easier for your case is to use the built in random.choices() function:

print(random.choices(Words, k=3))


Will print a list of 3 random words from your Words list.

If you aren't using Python 3.6, then you can just call random.choice(Words) over and over again.
"""
unseen_2 = clean_body(df['a_body'].iloc[100])
ans2 = summarize(unseen_2)
print(ans2)
print(unseen_2)
"""
pandas dataframe groupby and add the value of the value
#######################################################
I believe you need groupby:

df['D'] = df["C"].shift(1).groupby(df['A'], group_keys=False).rolling(2).mean()
print (df.head(20))
                   C     D
A     B                   
id 01 2018-01-01  10   NaN
      2018-01-02  11   NaN
      2018-01-03  12  10.5
      2018-01-04  13  11.5
      2018-01-05  14  12.5
      2018-01-06  15  13.5
      2018-01-07  16  14.5
      2018-01-08  17  15.5
      2018-01-09  18  16.5
      2018-01-10  19  17.5
id 02 2018-01-11  20   NaN
      2018-01-12  21  19.5
      2018-01-13  22  20.5
      2018-01-14  23  21.5
      2018-01-15  24  22.5
      2018-01-16  25  23.5
      2018-01-17  26  24.5
      2018-01-18  27  25.5
      2018-01-19  28  26.5
      2018-01-20  29  27.5


Or:

df['D'] = df["C"].groupby(df['A']).shift(1).rolling(2).mean()
print (df.head(20))
                   C     D
A     B                   
id 01 2018-01-01  10   NaN
      2018-01-02  11   NaN
      2018-01-03  12  10.5
      2018-01-04  13  11.5
      2018-01-05  14  12.5
      2018-01-06  15  13.5
      2018-01-07  16  14.5
      2018-01-08  17  15.5
      2018-01-09  18  16.5
      2018-01-10  19  17.5
id 02 2018-01-11  20   NaN
      2018-01-12  21   NaN
      2018-01-13  22  20.5
      2018-01-14  23  21.5
      2018-01-15  24  22.5
      2018-01-16  25  23.5
      2018-01-17  26  24.5
      2018-01-18  27  25.5
      2018-01-19  28  26.5
      2018-01-20  29  27.5
"""
unseen_3 = clean_body(df['a_body'].iloc[1000])
ans3 = summarize(unseen_3)
print(ans3)
print(unseen_3)
"""
how to get the value of a dataframe based on another column
###########################################################
Method 1: numpy.split &amp; DataFrame.loc:

We can split your columns into evenly size chunks and then use .loc to create the new columns:

for idx, chunk in enumerate(np.split(df.columns, len(df.columns)/4)):
    df[f'A{idx+1}_avg'] = df.loc[:, chunk].mean(axis=1)


Output

      A    B    C    D    E    F    G    H    I    J  ...    P    Q    R    S    T  A1_avg  A2_avg  A3_avg  A4_avg  A5_avg
0     0    1    2    3    4    5    6    7    8    9  ...   15   16   17   18   19     1.5     5.5     9.5    13.5    17.5
1    20   21   22   23   24   25   26   27   28   29  ...   35   36   37   38   39    21.5    25.5    29.5    33.5    37.5
2    40   41   42   43   44   45   46   47   48   49  ...   55   56   57   58   59    41.5    45.5    49.5    53.5    57.5
3    60   61   62   63   64   65   66   67   68   69  ...   75   76   77   78   79    61.5    65.5    69.5    73.5    77.5
4    80   81   82   83   84   85   86   87   88   89  ...   95   96   97   98   99    81.5    85.5    89.5    93.5    97.5
5   100  101  102  103  104  105  106  107  108  109  ...  115  116  117  118  119   101.5   105.5   109.5   113.5   117.5
6   120  121  122  123  124  125  126  127  128  129  ...  135  136  137  138  139   121.5   125.5   129.5   133.5   137.5
7   140  141  142  143  144  145  146  147  148  149  ...  155  156  157  158  159   141.5   145.5   149.5   153.5   157.5
8   160  161  162  163  164  165  166  167  168  169  ...  175  176  177  178  179   161.5   165.5   169.5   173.5   177.5
9   180  181  182  183  184  185  186  187  188  189  ...  195  196  197  198  199   181.5   185.5   189.5   193.5   197.5
10  200  201  202  203  204  205  206  207  208  209  ...  215  216  217  218  219   201.5   205.5   209.5   213.5   217.5
11  220  221  222  223  224  225  226  227  228  229  ...  235  236  237  238  239   221.5   225.5   229.5   233.5   237.5
12  240  241  242  243  244  245  246  247  248  249  ...  255  256  257  258  259   241.5   245.5   249.5   253.5   257.5
13  260  261  262  263  264  265  266  267  268  269  ...  275  276  277  278  279   261.5   265.5   269.5   273.5   277.5
14  280  281  282  283  284  285  286  287  288  289  ...  295  296  297  298  299   281.5   285.5   289.5   293.5   297.5
15  300  301  302  303  304  305  306  307  308  309  ...  315  316  317  318  319   301.5   305.5   309.5   313.5   317.5
16  320  321  322  323  324  325  326  327  328  329  ...  335  336  337  338  339   321.5   325.5   329.5   333.5   337.5
17  340  341  342  343  344  345  346  347  348  349  ...  355  356  357  358  359   341.5   345.5   349.5   353.5   357.5
18  360  361  362  363  364  365  366  367  368  369  ...  375  376  377  378  379   361.5   365.5   369.5   373.5   377.5
19  380  381  382  383  384  385  386  387  388  389  ...  395  396  397  398  399   381.5   385.5   389.5   393.5   397.5




Method 2: .range &amp; iloc:

We can create a range for each 4 columns, then use iloc to acces each slice of your dataframe and calculate the mean and at the same time create your new column:

slices = range(0, len(df.columns)+1, 4)

for idx, rng in enumerate(slices):
    if idx &gt; 0:
        df[f'A{idx}_avg'] = df.iloc[:, slices[idx-1]:slices[idx]].mean(axis=1)


Output

      A    B    C    D    E    F    G    H    I    J  ...    P    Q    R    S    T  A1_avg  A2_avg  A3_avg  A4_avg  A5_avg
0     0    1    2    3    4    5    6    7    8    9  ...   15   16   17   18   19     1.5     5.5     9.5    13.5    17.5
1    20   21   22   23   24   25   26   27   28   29  ...   35   36   37   38   39    21.5    25.5    29.5    33.5    37.5
2    40   41   42   43   44   45   46   47   48   49  ...   55   56   57   58   59    41.5    45.5    49.5    53.5    57.5
3    60   61   62   63   64   65   66   67   68   69  ...   75   76   77   78   79    61.5    65.5    69.5    73.5    77.5
4    80   81   82   83   84   85   86   87   88   89  ...   95   96   97   98   99    81.5    85.5    89.5    93.5    97.5
5   100  101  102  103  104  105  106  107  108  109  ...  115  116  117  118  119   101.5   105.5   109.5   113.5   117.5
6   120  121  122  123  124  125  126  127  128  129  ...  135  136  137  138  139   121.5   125.5   129.5   133.5   137.5
7   140  141  142  143  144  145  146  147  148  149  ...  155  156  157  158  159   141.5   145.5   149.5   153.5   157.5
8   160  161  162  163  164  165  166  167  168  169  ...  175  176  177  178  179   161.5   165.5   169.5   173.5   177.5
9   180  181  182  183  184  185  186  187  188  189  ...  195  196  197  198  199   181.5   185.5   189.5   193.5   197.5
10  200  201  202  203  204  205  206  207  208  209  ...  215  216  217  218  219   201.5   205.5   209.5   213.5   217.5
11  220  221  222  223  224  225  226  227  228  229  ...  235  236  237  238  239   221.5   225.5   229.5   233.5   237.5
12  240  241  242  243  244  245  246  247  248  249  ...  255  256  257  258  259   241.5   245.5   249.5   253.5   257.5
13  260  261  262  263  264  265  266  267  268  269  ...  275  276  277  278  279   261.5   265.5   269.5   273.5   277.5
14  280  281  282  283  284  285  286  287  288  289  ...  295  296  297  298  299   281.5   285.5   289.5   293.5   297.5
15  300  301  302  303  304  305  306  307  308  309  ...  315  316  317  318  319   301.5   305.5   309.5   313.5   317.5
16  320  321  322  323  324  325  326  327  328  329  ...  335  336  337  338  339   321.5   325.5   329.5   333.5   337.5
17  340  341  342  343  344  345  346  347  348  349  ...  355  356  357  358  359   341.5   345.5   349.5   353.5   357.5
18  360  361  362  363  364  365  366  367  368  369  ...  375  376  377  378  379   361.5   365.5   369.5   373.5   377.5
19  380  381  382  383  384  385  386  387  388  389  ...  395  396  397  398  399   381.5   385.5   389.5   393.5   397.5

[20 rows x 25 columns]
"""
#%% Save 
# tokenizers
with open('./saved_data/document_tokenizer.pickle', 'wb') as handle:
    pickle.dump(document_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./saved_data/summary_tokenizer.pickle', 'wb') as handle:
    pickle.dump(summary_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# transformer
transformer.save_weights('./saved_data/ts_checkpoints/transformer')

"""
# loading
with open('./saved_data/document_tokenizer.pickle', 'rb') as handle:
    document_tokenizer = pickle.load(handle)
with open('./saved_data/summary_tokenizer.pickle', 'rb') as handle:
    summary_tokenizer = pickle.load(handle)
"""


























