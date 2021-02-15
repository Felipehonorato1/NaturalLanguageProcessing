#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[15]:


def min_edit_distance(target,source):
    # Target word is the one we're trying to match from our source word.
    # In our matrix, the each caracter in our target word is going to be a row and our columns are going to be our source word
    
    target_list = [letter for letter in target]
    source_list = [letter for letter in source]
    
    matrix = np.zeros((len(source_list), len(target_list)))
    
    # First row, since we start with our </s> character 
    matrix[0] = [j for j in range(len(target_list))]
    matrix[:,0] = [j for j in range(len(source_list))]

    # Fill the array 
    for column in range(1, len(target_list)):
        for row in range(1, len(source_list)):
            if target_list[column] != source_list[row]:
                matrix[row,column]= min(matrix[row,column-1], matrix[row-1, column]) + 1
    
            else:
                matrix[row,column] = matrix[row-1, column-1]
    
    return matrix


# In[ ]:




