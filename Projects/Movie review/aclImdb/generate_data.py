#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import gensim
import os
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
from bs4 import BeautifulSoup


# In[12]:


class data_generator():
    
    def __init__(self): 
    
################################################################################################
    def generate_data(self,subset):
        start_time = time.time()
        # Making sure subset is either train or test
        assert subset == 'train' or subset == 'test'

        # These arrays are going to store our 
        pos_reviews = []
        neg_reviews = []

        pos_path = subset+'/pos'
        neg_path = subset+'/neg'


        # Generating both positive and negative arrays with its respective labels
        for folder in os.listdir(subset):

            if folder == 'pos':
                for file in os.listdir(pos_path):
                    file_path = pos_path+'/'+file
                    f = open(file_path,'r')

                    # Verifying if it could read the file properly
                    if(f):
                        pos_reviews.append(f.read())
                        f.close()

                pos_label_array = np.ones((len(pos_reviews)))

            else:
                for file in os.listdir(neg_path):
                    file_path = neg_path+'/'+file
                    f = open(file_path,'r')

                    # Verifying if it could read the file properly
                    if(f):
                        neg_reviews.append(f.read())
                        f.close()

                neg_label_array = np.zeros((len(neg_reviews)))

        # Joining arrays
        joint_reviews = np.concatenate((pos_reviews,neg_reviews))
        joint_labels = np.concatenate((pos_label_array,neg_label_array))

        # Making sure both have the same lenght
        assert len(joint_reviews) == len(joint_labels)
        print(f'Time took: {time.time() - start_time}s')
        return joint_reviews, joint_labels
################################################################################################
    
    def preprocessing_data(reviews_array,remove_sw= None):
    treated_array = []
    sw = stopwords.words('english')
    
    for review in reviews_array:
        
        #Removing extra whitespaces, lowering all cases in string and removing punctuation and stopwords
        review = review.lower()
        review = BeautifulSoup(review).get_text()
        review_no_punct = re.sub(r"[^\w\s]",'',review)
        review_tokens = word_tokenize(review_no_punct)
        
        if remove_sw:
            final_review = [word for word in review_tokens if word not in sw]
        else:
            final_review = review_tokens
            
        treated_array.append(" ".join(final_review).strip())
    
    assert len(reviews_array) == len(treated_array)
    return treated_array

################################################################################################

