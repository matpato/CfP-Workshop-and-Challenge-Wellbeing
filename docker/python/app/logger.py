import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


class Logger:
    def __init__(self):
        super().__init__()

    # Let's explore the contents of the dataset directory
    @staticmethod
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
        print(f"recall : {metrics.recall_score(expected_labels, predicted_labels, average='binary').__str__()}")
        print(f"f1 score : {metrics.f1_score(expected_labels, predicted_labels, average='weighted').__str__()}")
        print(f"precision : {metrics.precision_score(expected_labels, predicted_labels, average='weighted').__str__()}")
        # print(f"roc auc : {metrics.roc_auc_score(expected_labels, predicted_dataset, average='weighted',multi_dclass='ovo').__str__()}")
        print()
        print("############### ERROR ###############")
        print(f"mean squared error  : {metrics.mean_squared_error(expected_labels, predicted_labels).__str__()}")
        print(f"median error  : {metrics.median_absolute_error(expected_labels, predicted_labels).__str__()}")
        print(f"max error  : {metrics.max_error(expected_labels, predicted_labels).__str__()}")
        print()

    @staticmethod
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

    @staticmethod
    def logWorkDuration(startTime, endTime):
        print("\n")
        print("######################################")
        print("######### WORK DURATION INFO #########")
        print("######################################\n")
        print(f"started : {startTime.__str__()}")
        # Micro is more accurate, since the class distributions are not equal, as explained in the challenge
        print(f"ended : {endTime.__str__()}")
        print(f"duration : {endTime.__sub__(startTime).__str__()}")
        print()


class FeatureLogger :
    @staticmethod
    def logFeatureRankings(processedFeatures, labels):
        # Feature extraction
        model = LogisticRegression()
        rfe = RFE(model)
        fit = rfe.fit(processedFeatures, labels)
        print("Num Features: %s" % (fit.n_features_))
        print("Selected Features: %s" % (fit.support_))
        print("Feature Ranking: %s" % fit.ranking_)

    @staticmethod
    def logLabelCorrelation(processedFeatures, labels):
        # Feature extraction
        ridge = Ridge(alpha=0.00001)
        ridge.fit(processedFeatures, labels)
        Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
              normalize=False, random_state=None, solver='auto', tol=0.001)

        print("\n\nRidge model:\n", FeatureLogger._pretty_print_coefs(ridge.coef_))

    @staticmethod
    def logFeaturesCorrelation(processedFeatures,algorithm='pearson'):
        reversedTrainDataset = dict()
        for trainArrIndex in range(len(processedFeatures)):
            for featureTrainArrIndex in range(len(processedFeatures[trainArrIndex])):
                colName = 'f' + featureTrainArrIndex.__str__()
                if colName not in reversedTrainDataset: reversedTrainDataset[colName] = []
                reversedTrainDataset[colName].append(processedFeatures[trainArrIndex][featureTrainArrIndex])

        xyz = pd.DataFrame(reversedTrainDataset)

        corrMatrix = xyz.corr(method=algorithm).round(decimals=2)

        print(corrMatrix)

    def logFeaturesImportance(filteredDataset,filteredDatasetLabels):
        # define the model
        importanceModel = DecisionTreeRegressor()
        # fit the model
        importanceModel.fit(filteredDataset, filteredDatasetLabels)
        # get importance
        importance = importanceModel.feature_importances_
        importance = np.sort(importance)
        # summarize feature importance
        for i, v in enumerate(importance):
            # Useful to remove features that wont be used for the model
            print('Feature: %0d, Importance Score: %.5f' % (i, v))

    # A helper method for pretty-printing the coefficients
    @staticmethod
    def _pretty_print_coefs(coefs, names=None, sort=False):
        if names == None:
            names = ["X%s" % x for x in range(len(coefs))]
        lst = zip(coefs, names)
        if sort:
            lst = sorted(lst, key=lambda x: -np.abs(x[0]))
        return "\n".join("Feature %s - %s" % (name, round(coef, 3)) for coef, name in lst)