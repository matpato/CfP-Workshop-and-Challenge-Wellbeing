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
import datetime
import json
import os
import pandas as pd
import sys
import numpy as np
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
from sklearn import svm


# In[2]:


# <a id="1.2"></a>
# ### 1.2. Retrieve SMILE
# 
# The SMILE dataset was collected from 45 healthy adult participants (39 females and 6 males) in Belgium. The average age of participants was 24.5 years old, with a standard deviation of 3.0 years. Each participant contributed to an average of 8.7 days of data. Two types of wearable sensors were used for data collection. One was a wrist-worn device (Chillband, IMEC, Belgium) designed for the measurement of skin conductance (SC), ST, and acceleration data (ACC). The second sensor was a chest patch (Health Patch, IMEC, Belgium) to measure ECG and ACC. It contains a sensor node designed to monitor ECG at 256 Hz and ACC at 32 Hz continuously throughout the study period. Participants could remove the sensors while showering or before doing intense exercises. Also, participants received notifications on their mobile phones to report their momentary stress levels daily. 

# https://compwell.rice.edu/workshops/embc2022/dataset

# In[3]:
class ProcDataset:
    def __init__(self, dataArray, labels):
        self.dataArray = dataArray
        self.labels = labels

def logClassifer(clf):
    print("\n")
    print("######################################")
    print("########### CLASSIFIER INFO ##########")
    print("######################################\n")
    print(f"type - {clf.__str__()}\n")
    print(f"epsilon - {clf.epsilon.__str__()}")
    print(f"gamma - {clf.gamma.__str__()}")
    print(f"coef0 - {clf.coef0.__str__()}")
    print(f"kernel - {clf.kernel.__str__()}")
    print()

def logPredictionMetrics(expected_labels, predicted_labels):
    # Accuracy = TP+TN/TP+FP+FN+TN - accuracy
    # Recall = TP/TP+FN - true positive ratio
    # Precision = TP/TP+FP - all positive ratio
    # F1 Score = 2 * (Recall * Precision) / (Recall + Precision) - better for uneven classes

    print("\n")
    print("######################################")
    print("####### PREDICTION METRICS INFO ######")
    print("######################################\n")

    print(f"accuracy : {metrics.accuracy_score(expected_labels, predicted_labels).__str__()}")
    # Micro is more accurate, since the class distributions are not equal, as explained in the challenge
    print(f"recall : {metrics.recall_score(expected_labels, predicted_labels, average='weighted').__str__()}")
    print(f"f1 score : {metrics.f1_score(expected_labels, predicted_labels, average='weighted').__str__()}")
    print(f"precision : {metrics.precision_score(expected_labels, predicted_labels, average='weighted').__str__()}")
    #print(f"roc auc : {metrics.roc_auc_score(expected_labels, predicted_dataset, average='weighted',multi_dclass='ovo').__str__()}")
    print()
    print("############### ERROR ###############")
    print(f"mean squared error  : {metrics.mean_squared_error(expected_labels, predicted_labels).__str__()}")
    print(f"median error  : {metrics.median_absolute_error(expected_labels, predicted_labels).__str__()}")
    print(f"max error  : {metrics.max_error(expected_labels, predicted_labels).__str__()}")
    print()

def logWorkDuration(startTime,endTime):
    print("\n")
    print("######################################")
    print("######### WORK DURATION INFO #########")
    print("######################################\n")
    print(f"started : {startTime.__str__()}")
    # Micro is more accurate, since the class distributions are not equal, as explained in the challenge
    print(f"ended : {endTime.__str__()}")
    print(f"duration : {endTime.__sub__(startTime).__str__()}")
    print()

def processDataset(smallSataset, featureStr, dataStr, removeNan=False):
    nfeatures = len(smallSataset[featureStr][dataStr][0][0])
    n = len(smallSataset[featureStr][dataStr]) * \
        len(smallSataset[featureStr][dataStr][0])
    # variaveis e iniciação a zero
    dataset_features = np.zeros((n, nfeatures))  # X
    dataset_labels = np.zeros(n)  # y
    #
    count = 0
    for i in range(len(smallSataset[featureStr][dataStr])):
        for j in range(len(smallSataset[featureStr][dataStr][i])):
            if not removeNan :
                dataset_features[count, 0:nfeatures] = smallSataset[featureStr][dataStr][i][j]
                dataset_labels[count] = smallSataset['labels'][i]
                count = count + 1
            elif (np.sum(np.isnan(smallSataset[featureStr][dataStr][i][j])) == 0):
                # nao considerar os nan
                dataset_features[count,0:nfeatures] = smallSataset[featureStr][dataStr][i][j]
                dataset_labels[count] = smallSataset['labels'][i]
                count = count + 1

    return ProcDataset(dataset_features[0:count],dataset_labels[0:count])


def experimentClassifier(clf,dataset, datasetLabels, nrOfIters=1):
    sgkf = model_selection.StratifiedKFold(n_splits=4, shuffle=True)

    for trainIndexes, testIndexes in sgkf.split(dataset, datasetLabels):
        kfold_train_dataset = trainDataset[trainIndexes]
        kfold_train_dataset_labels = procDataTrainHandcraftedHR.labels[trainIndexes]
        kfold_test_dataset = trainDataset[testIndexes]
        kfold_test_dataset_labels = procDataTrainHandcraftedHR.labels[testIndexes]

        clf.fit(kfold_train_dataset, kfold_train_dataset_labels)
        kfold_predicted_labels = clf.predict(kfold_test_dataset)
        logPredictionMetrics(kfold_test_dataset_labels, kfold_predicted_labels)

        nrOfIters = nrOfIters-1
        if nrOfIters == 0:
            break

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
startTime = datetime.datetime.now().replace(microsecond=0)

dataset = np.load('./dataset_smile_challenge.npy', allow_pickle=True).item()

dataset_train = dataset['train']

dataset_test = dataset['test']

procDataTrainHandcraftedHR = processDataset(dataset_train, 'hand_crafted_features', 'ECG_features', removeNan=True)
procDataTestHandcraftedHR = processDataset(dataset_test, 'hand_crafted_features', 'ECG_features', removeNan=True)



# Now we have a DataFrame with the contents of the metadata file.

# <a id="4"></a>
# ## 4. Data statistics

# In[9]:


print(
    f"train: {dataset['train']['hand_crafted_features']['ECG_features'].shape}")
print(
    f"test: {dataset['test']['hand_crafted_features']['ECG_features'].shape}")

# ### Classifier

# In[17]:


startTrainIndex = 0
endTrainIndex = 20000#len(procDataTrainHandcraftedHR.dataArray)

trainDataset = procDataTrainHandcraftedHR.dataArray[startTrainIndex:endTrainIndex]

clf = svm.SVC()
experimentClassifier(clf, trainDataset[startTrainIndex:endTrainIndex], procDataTrainHandcraftedHR.labels[startTrainIndex:endTrainIndex])






#startTestIndex = 0
#endTestIndex = 20000#len(procDataTestHandcraftedHR.dataArray)

#testDataset = procDataTestHandcraftedHR.dataArray[startTestIndex:endTestIndex]

#predicted_labels = clf.predict(testDataset)
#expected_labels = procDataTestHandcraftedHR.labels[startTestIndex:endTestIndex]


# In[3]:
endTime = datetime.datetime.now().replace(microsecond=0)
logWorkDuration(startTime,endTime)
logClassifer(clf)
