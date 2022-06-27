
import numpy as np
import numpy.random.mtrand
from scipy.stats import pearsonr
from sklearn import model_selection, metrics, tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from logger import Logger


class FilteredDataset:
    def __init__(self, ecgFeatures, gsrFeatures,labels):
        self.ecgFeatures = ecgFeatures
        self.gsrFeatures = gsrFeatures
        self.labels = labels

#Returns an array with the arrays to keep in all major arrays (labels and features)
def filterZerosHandCraftedFeatures(handcraftedFeaturesDataset, handcraftedFeaturesLabels,filterZeros=True):

    #retornar array com indexes a manter ? (Masi facil)
    indexesToRemove = []

    for key in handcraftedFeaturesDataset.keys():
        if "masking" not in key: continue
        if not filterZeros: continue
        for i in range(len(handcraftedFeaturesDataset[key])) :
            if np.min(handcraftedFeaturesDataset[key][i]) == 0:
                indexesToRemove.append(i)

    indexesToKeep = range(len(handcraftedFeaturesLabels))
    #else: indexesToKeep = np.delete(range(len(handcraftedFeaturesLabels)), np.unique(indexesToRemove))

    return FilteredDataset(ecgFeatures=handcraftedFeaturesDataset['ECG_features'][indexesToKeep],
                           gsrFeatures=handcraftedFeaturesDataset['GSR_features'][indexesToKeep],
                           labels=handcraftedFeaturesLabels[indexesToKeep])


def discretizeHandCraftedDataset(processedHandTrainDataset):
    n_entries = len(processedHandTrainDataset.labels)
    nOfFeatures= 6
    dataset_features = np.zeros((n_entries, nOfFeatures))
    #Array positions
    # 0 - mean minute hr
    # 0 - mean gsr
    #
    #
    #
    #
    #

    for index_n_entries in range(n_entries):
        dataset_features[index_n_entries] = [

            ####### HR Features
            np.nanmean([i[0] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]), ## FIX THE OTHERS TO BE LIKE THIS
            #np.nansum([i[1] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),
            #np.nanvar([i[2] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),
            np.nansum([i[2] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),
            np.nanpercentile([i[2] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]], q=75),  # KEEP -0.20
            #np.nanpercentile(processedHandTrainDataset.ecgFeatures[index_n_entries][0], q=10),  # KEEP -0.20
            #np.nanpercentile(processedHandTrainDataset.ecgFeatures[index_n_entries][0], q=5),  # KEEP 0.23

            ######## GSR Features
            np.median([i[0] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            #np.var([i[0] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            #np.percentile(processedHandTrainDataset.gsrFeatures[index_n_entries][0], q=15),
            ######## ST Features
            #np.nanvar([i[9] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            #np.var([i[8] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            np.var([i[8] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            np.max([i[9] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]) -
            np.min([i[9] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            #np.max([i[10] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]])
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



def experimentClassifier(clf, dataset, datasetLabels, nrOfIters=1, binarizeOutputLabels=False, binarizeAllLabels=False):
    sgkf = model_selection.StratifiedKFold(n_splits=2, shuffle=True,random_state=numpy.random.mtrand.randint(low=0,high=10000))

    for trainIndexes, testIndexes in sgkf.split(dataset, datasetLabels):
        kfold_train_dataset = dataset[trainIndexes]
        kfold_train_dataset_labels = datasetLabels[trainIndexes]
        kfold_test_dataset = dataset[testIndexes]
        kfold_test_dataset_labels = datasetLabels[testIndexes]

        if binarizeAllLabels :
            kfold_train_dataset_labels = np.where(kfold_train_dataset_labels==0,0,1)
            kfold_test_dataset_labels = np.where(kfold_test_dataset_labels==0,0,1)



        clf.fit(kfold_train_dataset, kfold_train_dataset_labels)
        kfold_predicted_labels = clf.predict(kfold_test_dataset)

        #Printing results
        if binarizeOutputLabels or binarizeAllLabels:
            Logger.logPredictionMetrics(np.where(kfold_test_dataset_labels==0,0,1), np.where(kfold_predicted_labels==0,0,1))
        else :
            Logger.logPredictionMetrics(kfold_test_dataset_labels,kfold_predicted_labels)

        nrOfIters = nrOfIters-1
        if nrOfIters == 0:
            break


def fitAndPredictClassifier(clf,trainDataset, trainDatasetLabels, testDataset, testDatasetLabels=None,binarizeLabels=True, printMetrics=False, saveOutputFile=True):
    sgkf = model_selection.StratifiedKFold(n_splits=2, shuffle=True)

    for trainIndexes, testIndexes in sgkf.split(trainDataset, trainDatasetLabels):
        kfold_train_dataset = trainDataset[trainIndexes]
        kfold_train_dataset_labels = trainDatasetLabels[trainIndexes]

        clf.fit(kfold_train_dataset, kfold_train_dataset_labels)
        break

    clf.fit(trainDataset, trainDatasetLabels)
    predicted_labels = clf.predict(testDataset)

    if printMetrics :
        Logger.logPredictionMetrics(testDatasetLabels, predicted_labels)

    if binarizeLabels :
        predicted_labels = np.where(predicted_labels==0,0,1)

    if saveOutputFile :
        np.savetxt('data/answer.txt',predicted_labels, fmt='%i')

# In[4]:

dataset = np.load('data/dataset_smile_challenge.npy', allow_pickle=True).item()

dataset_train_handcrafted_features = dataset['train']['hand_crafted_features']
dataset_train_labels = dataset['train']['labels']

dataset_train_ecg_features = dataset['train']['hand_crafted_features']['ECG_features']
dataset_train_gsr_features = dataset['train']['hand_crafted_features']['ECG_features']

dataset_test = dataset['test']

##### PROCESS DATA

processedHandTrainDataset=filterZerosHandCraftedFeatures(dataset_train_handcrafted_features,dataset_train_labels)

trainDatasetTest = discretizeHandCraftedDataset(processedHandTrainDataset)

print('0.6-0-58 de precisao deu 0.547 no real, so you need higher')






startTrainIndex = 0
endTrainIndex = len(trainDatasetTest)

trainDataset = trainDatasetTest[startTrainIndex:endTrainIndex]
trainDatasetLabels = processedHandTrainDataset.labels[startTrainIndex:endTrainIndex]


#clf = tree.DecisionTreeClassifier(class_weight=[{0: 1, 1: 5}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 0}])
#clf = tree.DecisionTreeClassifier(class_weight={0: 2, 1: 1, 2: 2, 3: 3}) #0.54

#clf = RandomForestClassifier(max_depth=18,max_features='log2')




model = LinearRegression()
# fit the model
model.fit(trainDataset, trainDatasetLabels)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))

for featureLength in range(len(trainDataset[0])) :
    corr, _ = pearsonr([i[featureLength] for i in trainDataset], trainDatasetLabels)
    print('Pearsons correlation: %.3f' % corr)

#exit(0)

clf = RandomForestClassifier(max_depth=10,max_features='log2')

experimentClassifier(clf, trainDataset, trainDatasetLabels,nrOfIters=3, binarizeAllLabels=True)


####################################################
####################################################
#                   TESTING
####################################################
####################################################

dataset_test_handcrafted_features = dataset['test']['hand_crafted_features']
dataset_test_labels = dataset['test']['labels']
processedHandTestDataset=filterZerosHandCraftedFeatures(dataset_test_handcrafted_features,dataset_test_labels,filterZeros=False)

testDatasetTest = discretizeHandCraftedDataset(processedHandTestDataset)

fitAndPredictClassifier(clf, trainDataset, trainDatasetLabels, testDatasetTest)






