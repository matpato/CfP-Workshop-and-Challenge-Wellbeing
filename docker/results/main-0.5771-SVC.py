
import numpy as np
import numpy.random.mtrand
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

from logger import Logger, FeatureLogger


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
    nOfFeatures= 4
    dataset_features = np.zeros((n_entries, nOfFeatures))
    #Array positions
    # 0 - mean minute hr
    # 0 - mean gsr

    for index_n_entries in range(n_entries):
        dataset_features[index_n_entries] = [

            ####### HR Features
            np.nanmean([i[0] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),  ## KEEP
            np.nanmean([i[0] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),  ## KEEP
            np.nanmean([i[0] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]), ## KEEP
            #np.count_nonzero([i[2]>0.5 for i in processedHandTrainDataset.ecgFeatures[index_n_entries]])/len([i[2] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]]),
            np.nanpercentile([i[7] for i in processedHandTrainDataset.ecgFeatures[index_n_entries]], q=80),
            ######## GSR Features
            #np.nanvar([i[4] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            # -0.009, covariance :-0.000
            ######## ST Features
            #np.nanmean([i[8] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]),
            #np.nanstd([i[10] for i in processedHandTrainDataset.gsrFeatures[index_n_entries]]), #KEEP


         ]

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



def experimentClassifier(clf, dataset, datasetLabels, nrOfIters=1, binarizeOutputLabels=False, binarizeAllLabels=False):
    sgkf = model_selection.StratifiedKFold(n_splits=3, shuffle=False)

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
    sgkf = model_selection.StratifiedKFold(n_splits=2,shuffle=False,)

    #clf.fit(trainDataset, trainDatasetLabels)
    predicted_labels = clf.predict(testDataset)

    if printMetrics :
        Logger.logPredictionMetrics(testDatasetLabels, predicted_labels)

    if binarizeLabels :
        predicted_labels = np.where(predicted_labels==0,0,1)

    if saveOutputFile :
        np.savetxt('data/answer.txt',predicted_labels, fmt='%i')

def reg_m(x, y):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, x).fit()
    return results

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

# In[4]:

dataset = np.load('data/dataset_smile_challenge.npy', allow_pickle=True).item()

dataset_train_handcrafted_features = dataset['train']['hand_crafted_features']
dataset_train_labels = dataset['train']['labels']

dataset_handcrafted_masking = dataset['train']['hand_crafted_features']['ECG_masking']
dataset_deep_masking = dataset['train']['deep_features']['masking']

dataset_test = dataset['test']

##### PROCESS DATA

processedHandTrainDataset=filterZerosHandCraftedFeatures(dataset_train_handcrafted_features,dataset_train_labels)

trainDatasetTest = discretizeHandCraftedDataset(processedHandTrainDataset)

print('0.6-0-58 de precisao deu 0.547 no real, so you need higher')






startTrainIndex = 0
endTrainIndex = len(trainDatasetTest)

trainDataset = trainDatasetTest[startTrainIndex:endTrainIndex]
trainDatasetLabels = np.where(processedHandTrainDataset.labels[startTrainIndex:endTrainIndex]==0,0,1)

trainDataset, trainDatasetLabels= balanced_subsample(trainDataset,trainDatasetLabels)


#clf = tree.DecisionTreeClassifier(class_weight=[{0: 1, 1: 5}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 0}])
#clf = tree.DecisionTreeClassifier(class_weight={0: 2, 1: 1, 2: 2, 3: 3}) #0.54

#clf = RandomForestClassifier(max_depth=18,max_features='log2')

# define the model
importanceModel = DecisionTreeRegressor()
# fit the model
importanceModel.fit(trainDataset, trainDatasetLabels)
# get importance
importance = importanceModel.feature_importances_
importance=np.sort(importance)
# summarize feature importance
for i,v in enumerate(importance):
    #Useful to remove features that wont be used for the model
    print('Feature: %0d, Importance Score: %.5f' % (i, v))

#for featureIndex in range(len(trainDataset[0])) :
#corr, _ = pearsonr(dataset_handcrafted_masking.flatten() , dataset_deep_masking.flatten())
#covariance= np.cov(dataset_handcrafted_masking.flatten(), dataset_deep_masking.flatten())[0][1]
#print('Feature: %0d, Pearsons correlation: %.3f, covariance :%.3f' % (0, corr,covariance))

#for featureIndex in range(len(trainDataset[0])) :
#    corr, _ = pearsonr([i[featureIndex] for i in trainDataset], binarizedTrainLabels)
#    covariance= np.cov([i[featureIndex] for i in trainDataset], binarizedTrainLabels)[0][1]
#    print('Feature: %0d, Pearsons correlation: %.3f, covariance :%.3f' % (featureIndex, corr,covariance))
#print(reg_m(trainDataset, trainDatasetLabels).summary())

FeatureLogger.logFeatureRankings(trainDataset,trainDatasetLabels)
FeatureLogger.logCorrelations(trainDataset,trainDatasetLabels)

#exit(0)

#clf = RandomForestClassifier(min_samples_leaf=10,max_depth=10, max_features=None) #BEST CLASSIFIER SO FAR - 0.45
clf = svm.LinearSVC(class_weight={0:1,1:1,3:0,4:0}) # 0.53


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






