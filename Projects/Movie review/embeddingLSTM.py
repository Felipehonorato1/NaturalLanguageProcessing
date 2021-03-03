#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, Activation,TimeDistributed
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.preprocessing.sequence import pad_sequences


# In[3]:


def embedding_lstm_model(input_shape,vocab_size,n_outputs):
    
    lr = 1e-3
    
    inputs = Input(shape = input_shape)
    embedding_layer = Embedding(vocab_size, output_dim = 256)
    rnn_layer = LSTM(units = 256, return_sequences = True)(embedding_layer)
    logits = TimeDistributed(Dense(n_outputs))(rnn_layer)
    outputs = Activation('softmax')(logits)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = Adam(lr), metrics = ['accuracy'], loss = sparse_categorical_crossentropy )
    
    
    return model


# In[ ]:




