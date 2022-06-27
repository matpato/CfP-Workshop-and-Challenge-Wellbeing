
import numpy as np
import numpy.random.mtrand
import pandas as pd
from scipy.stats import pearsonr
from sklearn import model_selection, metrics, tree
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, IsolationForest
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from dataset_process import DatasetOperations
from logger import Logger, FeatureLogger







def discretizeHandCraftedDataset(processedHandTrainDataset):
    n_entries = len(processedHandTrainDataset.labels)
    nOfFeatures=5
    dataset_features = np.zeros((n_entries, nOfFeatures))
    #Array positions
    # 0 - mean minute hr
    # 0 - mean gsr

    for index_n_entries in range(n_entries):
        dataset_features[index_n_entries] = [

            ####### HR Features
            np.nanmean([i[0] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),  ## 0.20 , 0.006
            # np.nanpercentile([i[1] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]], q=80),
            # np.nanquantile([i[2] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]],q=0.4),
            # np.nansum([i[3] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),
            np.nanmean([i[7] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),
            # np.nanquantile([i[5] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]], q=0.90),
            ## 0.156 , 0.001

            # np.nanpercentile([i[4] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]], q=85), ## 0.148 , 0.014

            # np.nanpercentile([i[6] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]], q=10), #0.145, covariance :0.003

            # np.nansum([i[4] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]), # 0.153 , 0.938

            ######## GSR Features
            np.nanvar([i[4] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            np.nanquantile([i[5] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]],
                           q=0.85) - np.nanquantile(
                [i[0] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]], q=0.15),
            # -0.009, covariance :-0.000
            # np.nanvar([i[4] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            # -0.009, covariance :-0.000
            ######## ST Features
            np.nanquantile([i[10] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]], q=0.95),
        ]


    return dataset_features

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



def experimentClassifier(clf, dataset, datasetLabels, nIters=1, binarizeOutputLabels=False, binarizeAllLabels=False):
    sgkf = model_selection.StratifiedKFold(n_splits=3, shuffle=False)

    for trainIndexes, testIndexes in sgkf.split(dataset, datasetLabels):
        kfold_train_dataset = dataset[trainIndexes]
        kfold_train_dataset_labels = datasetLabels[trainIndexes]
        kfold_test_dataset = dataset[testIndexes]
        kfold_test_dataset_labels = datasetLabels[testIndexes]

        if binarizeAllLabels :
            kfold_train_dataset_labels = DatasetOperations.binarizeLabels(kfold_train_dataset_labels)
            kfold_test_dataset_labels =DatasetOperations.binarizeLabels(kfold_test_dataset_labels)



        clf.fit(kfold_train_dataset, kfold_train_dataset_labels)
        kfold_predicted_labels = clf.predict(kfold_test_dataset)

        #Printing results
        if binarizeOutputLabels or binarizeAllLabels:
            Logger.logPredictionMetrics(np.where(kfold_test_dataset_labels==0,0,1), np.where(kfold_predicted_labels==0,0,1))
        else :
            Logger.logPredictionMetrics(kfold_test_dataset_labels,kfold_predicted_labels)

        nIters = nIters - 1
        if nIters == 0:
            break



def fitAndPredictClassifier(clf,trainDataset, trainDatasetLabels, testDataset, testDatasetLabels=None,binarizeLabels=True, printMetrics=False, saveOutputFile=True):
    sgkf = model_selection.StratifiedKFold(n_splits=2,shuffle=False,)

    #clf.fit(trainDataset, trainDatasetLabels)
    predicted_labels = clf.predict(testDataset)

    if printMetrics :
        Logger.logPredictionMetrics(testDatasetLabels, predicted_labels)

    if binarizeLabels :
        predicted_labels = np.where(predicted_labels==0,0,1)

    if saveOutputFile :
        np.savetxt('data/answer.txt',predicted_labels, fmt='%i')





# LOAD AND ORGANIZE DATA:

dataset = np.load('data/dataset_smile_challenge.npy', allow_pickle=True).item()

dataset_train_handcrafted_features = dataset['train']['hand_crafted_features']
dataset_train_labels = dataset['train']['labels']

dataset_handcrafted_masking = dataset['train']['hand_crafted_features']['ECG_masking']
dataset_deep_masking = dataset['train']['deep_features']['masking']

dataset_test = dataset['test']

##### PROCESS DATA

processedHandTrainDataset=DatasetOperations.filterZerosHandCraftedFeatures(dataset_train_handcrafted_features,dataset_train_labels,filterZeros=True)
trainDatasetTest = discretizeHandCraftedDataset(processedHandTrainDataset)

startTrainIndex = 0
endTrainIndex = len(trainDatasetTest)

trainDataset = trainDatasetTest[startTrainIndex:endTrainIndex]
trainDatasetLabels = DatasetOperations.binarizeLabels(dataset_train_labels)

trainDataset, trainDatasetLabels= DatasetOperations.balancedSubsample(trainDataset, trainDatasetLabels)


####################################################
####################################################
#                   TRAINING
####################################################
####################################################

#LOGS
FeatureLogger.logFeatureRankings(trainDataset,trainDatasetLabels)
FeatureLogger.logLabelCorrelation(trainDataset, trainDatasetLabels)
FeatureLogger.logFeaturesCorrelation(trainDataset)

#exit(0)

#clf = RandomForestClassifier(max_features='log2') #BEST CLASSIFIER SO FAR - 0.45
#clf = tree.DecisionTreeClassifier(class_weight={0: 2, 1: 1, 2: 2, 3: 3}) #0.54

#clf = RandomForestClassifier(max_depth=10,max_features=None)
clf = LinearSVC()

experimentClassifier(clf, trainDataset, trainDatasetLabels, nIters=3, binarizeAllLabels=True)


####################################################
####################################################
#                   TESTING
####################################################
####################################################

dataset_test_handcrafted_features = dataset_test['hand_crafted_features']
dataset_test_labels = dataset_test['labels']
processedHandTestDataset=DatasetOperations.createFilteredDataset(dataset_test_handcrafted_features,dataset_test_labels)

testDatasetTest = discretizeHandCraftedDataset(processedHandTestDataset)

fitAndPredictClassifier(clf, trainDataset, trainDatasetLabels, testDatasetTest)
