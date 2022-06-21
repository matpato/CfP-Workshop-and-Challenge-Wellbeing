#!/usr/bin/env python
# coding: utf-8

# # Challenge: Workshop and Challenge on Detection of Stress and Mental Health Using Wearable Sensors
# 
# <ul>
#     <li><a href="#1">1. Data retrieval and cleaning</a></li>
# </ul>
#    
# <ul>
#    <li>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.1">1.1.Import libraries</a></li>
# </ul>
# 
# <ul>
#    <li>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.2">1.2. Retrieve dataset</a></li>
# </ul>
# <ul>
#    <li>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.4">1.3. Selecting a sample</a></li>
# </ul>
# 
# 
# <ul>
#    <li><a href="#4">4. Data statistics</a></li>
# </ul>
# 

# <a id="1"></a>
# 
# ## 1. Data retrieval + cleaning

# <a id="1.1"></a>
# ### 1.1 Import libraries

# In[1]:


import json
import os
import pandas as pd
import requests
import sys
import numpy as np


# In[2]:


pip install --upgrade pip


# <a id="1.2"></a>
# ### 1.2. Retrieve SMILE
# 
# The SMILE dataset was collected from 45 healthy adult participants (39 females and 6 males) in Belgium. The average age of participants was 24.5 years old, with a standard deviation of 3.0 years. Each participant contributed to an average of 8.7 days of data. Two types of wearable sensors were used for data collection. One was a wrist-worn device (Chillband, IMEC, Belgium) designed for the measurement of skin conductance (SC), ST, and acceleration data (ACC). The second sensor was a chest patch (Health Patch, IMEC, Belgium) to measure ECG and ACC. It contains a sensor node designed to monitor ECG at 256 Hz and ACC at 32 Hz continuously throughout the study period. Participants could remove the sensors while showering or before doing intense exercises. Also, participants received notifications on their mobile phones to report their momentary stress levels daily. 

# https://compwell.rice.edu/workshops/embc2022/dataset

# In[3]:


dataset = np.load('data/dataset_smile_challenge.npy', allow_pickle=True).item()


#     dict
#         dictionary with dataset, with keys:
#          * `train`
#           * `deep_features`
#            * `ECG_features_C`
#            * `ECG_features_T`
#            * `masking`
#           * `hand_crafted_features`
#            * `ECG_features`
#            * `ECG_masking`
#            * `GSR_features`
#            * `GSR_masking`
#           * `labels`
#          * `test`
#           * Same structure as `train`

# Let's explore the contents of the dataset directory

# In[4]:


# for training and testing data:

dataset_train = dataset['train']

dataset_test = dataset['test']

# for deep features.

deep_features = dataset_train['deep_features']

# conv1d backbone based features for ECG signal.

deep_features['ECG_features_C'] 

# transformer backbone basde features for ECG signal  

deep_features['ECG_features_T']   

# for hand-crafted features.

handcrafted_features_train = dataset_train['hand_crafted_features']
handcrafted_features_test = dataset_test['hand_crafted_features']

# handcrafted features for ECG signal

handcrafted_features_train['ECG_features'] 
handcrafted_features_test['ECG_features']

 # handcrafted features for GSR signal. 

handcrafted_features_train['GSR_features'] 
handcrafted_features_test['GSR_features']

# for labels.

labels_train = dataset_train['labels']  # labels.
labels_test = dataset_test['labels']


# In[5]:


dataset['test'].keys()


# In[82]:


len(dataset['train']['labels'])


# Now we have a DataFrame with the contents of the metadata file.

# <a id="4"></a>
# ## 4. Data statistics 

# In[9]:


print(
    f"train: {dataset['train']['hand_crafted_features']['ECG_features'].shape}")
print(
    f"test: {dataset['test']['hand_crafted_features']['ECG_features'].shape}")


# Load SMILE dataset as a dictionary from npy file.
# Each feature matrix has 3 dimensions:
# * sequence (of 60 minutes)
# * window (5 minute with 4 min overlap)
# * feature

# ## ECG Features
# * feature and label vector construction
# * creation of classifier

# In[8]:


len(dataset['train']['hand_crafted_features']['ECG_features'])


# In[6]:


dataset['train']['hand_crafted_features']['ECG_features'][0].shape #tem uma sequencia de 60 minutos ?


# In[85]:


# representa as features extraidas de 1 janela da sequencia (correspondente a 5 minutos)
dataset['test']['hand_crafted_features']['ECG_features'][0][0]


# In[34]:


# Create an array with features
# Train dataset
nfeatures = len(dataset['train']['hand_crafted_features']['ECG_features'][0][0])
n = len(dataset['train']['hand_crafted_features']['ECG_features']) * \
    len(dataset['train']['hand_crafted_features']['ECG_features'])
# variaveis e iniciação a zero
handcrafted_features_train=np.zeros((n,nfeatures)) #X
handcrafted_labels_train = np.zeros(n)  # y
#
count=0
for i in range(len(dataset['train']['hand_crafted_features']['ECG_features'])):
    for j in range(len(dataset['train']['hand_crafted_features']['ECG_features'][i])):
        if(np.sum(np.isnan(dataset['train']['hand_crafted_features']['ECG_features'][i][j])) == 0):
            # nao considerar os nan
            handcrafted_features_train[count,
                                       0:nfeatures] = dataset['train']['hand_crafted_features']['ECG_features'][i][j]
            handcrafted_labels_train[count] = dataset['train']['labels'][i]
            count=count+1      


# In[ ]:


# Test dataset
nfeatures = len(dataset['test']['hand_crafted_features']['ECG_features'][0][0])
n = len(dataset['test']['hand_crafted_features']['ECG_features']) * \
    len(dataset['test']['hand_crafted_features']['ECG_features'][0])
# variaveis e iniciação a zero
handcrafted_features_test=np.zeros((n,nfeatures)) #X
handcrafted_labels_test = np.zeros(n)  # y

count=0
for i in range(len(dataset['test']['hand_crafted_features']['ECG_features'])):
    #print(dataset['test']['hand_crafted_features']['ECG_features'][i])
    for j in range(len(dataset['test']['hand_crafted_features']['ECG_features'][i])):
        if(np.sum(np.isnan(dataset['test']['hand_crafted_features']['ECG_features'][i][j])) == 0):
            # nao considerar os nan
            handcrafted_features_test[count,
                                      0:nfeatures] = dataset['test']['hand_crafted_features']['ECG_features'][i][j]
            handcrafted_labels_test[count] = dataset['test']['labels'][i]
            count = count+1


# In[41]:


# Remove all rows with zeros from the array
data1 = handcrafted_features_train
data1[~np.all(data1 == 0, axis=1)].shape

# Remove last values from the array
data_train = handcrafted_features_train
data_train[:-(n-count),:].shape  

data_test = handcrafted_features_test
data_test[:-(n-count),:].shape


# In[42]:


## Labels
data_train_label = handcrafted_labels_train
data_train_label[:-(n-count)].shape

data_test_label = handcrafted_labels_test
data_test_label[:-(n-count)].shape


# In[37]:


## find max
np.where(data_train_label == np.max(data_train_label))


# In[38]:


## find min
np.where(data_train_label==np.min(data_train_label))


# In[39]:


## find uniques
np.unique(data_train_label)


# In[45]:


# Convert to a dataframe and save it in csv
df_train = pd.DataFrame(data=np.column_stack((data_train, data_train_label)))
df_train.columns = ["F"+str(i) for i in range(1, len(df_train.columns) + 1)]
df_train.rename(columns={'F9': 'Label'}, inplace=True)
df_train


# In[43]:


# Convert to a dataframe and save it in csv
df_test = pd.DataFrame(data=np.column_stack((data_test, data_test_label)))
df_test.columns = ["F"+str(i) for i in range(1, len(df_test.columns) + 1)]
df_test.rename(columns={'F9': 'Label'}, inplace=True)
df_test


# In[46]:


df_train.to_csv('data/data_train.csv')
df_test.to_csv('data/data_test.csv')


# ### Classifier

# In[17]:


from sklearn import svm
clf = svm.SVC()
clf.fit(data_train, data_train_label)


# In[47]:


dataset['train']['hand_crafted_features']['GSR_features'].shape


# In[48]:


dataset['test']['hand_crafted_features']['GSR_features'].shape

