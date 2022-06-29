import numpy as np
import pandas as pd
from sklearn import model_selection, metrics
import tensorflow as tf
from keras import models, layers
from keras.layers import LSTM
from tensorflow.python.keras import Input

class ClfMetrics:
    def __init__(self, accuracy, f1score):
        self.accuracy = accuracy
        self.f1score = f1score

    def isBetter(self, newMetric, f1ScoreTreshold = 0.8):
        isAccuracyBetter = newMetric.accuracy > self.accuracy
        isF1ScoreBetter = newMetric.f1score > self.f1score * f1ScoreTreshold
        return isAccuracyBetter and isF1ScoreBetter

class FeatureSelector :
    def __init__(self,featuresDataset,featuresDatasetLabels):
        self.featuresDataset = featuresDataset
        self.featuresDatasetLabels = featuresDatasetLabels
        self.currentMetrics = ClfMetrics(0, 0)
        self.featuresIndexes = np.arange(0, len(self.featuresDataset[0]))



    def selectFeatures(self,clf, minFeatures = 3, maxFeatures = 5,maxf1ScoreDiff=0.8):

        self.featuresIndexes = np.arange(0, len(self.featuresDataset[0]))

        pickedFeaturesIndexes =self.featuresIndexes

        self.currentMetrics = self.testModel(clf, self.featuresIndexes)

        maxFeatureCorrelation = 0.6

        while 1 :

            pickedFeaturesIndexes = self._removeCorrelatedFeatures(clf,pickedFeaturesIndexes,maxFeatureCorrelation)
            print('hey1')

            newPickedFeatures = self._tryRemoveFeatures(clf,pickedFeaturesIndexes,maxf1ScoreDiff)
            print('hey2')

            if len(pickedFeaturesIndexes)> len(newPickedFeatures) and minFeatures < len(pickedFeaturesIndexes):
                pickedFeaturesIndexes = newPickedFeatures
            elif minFeatures >= len(pickedFeaturesIndexes) or len(pickedFeaturesIndexes) == len(newPickedFeatures):
                break




        return pickedFeaturesIndexes

    def removeCorrelatedFeatures (self,clf,pickedFeatures,maxFeatureCorrelation) :
        featureMatrix = self.calculateFeaturesCorrelation(np.take(self.featuresDataset,pickedFeatures,axis=1))
        for featureMatrixX in range(len(featureMatrix.values)):
            for featureMatrixY in range(len(featureMatrix.values[featureMatrixX])):
                if featureMatrix.values[featureMatrixX][featureMatrixY] == 1:
                    continue
                elif featureMatrix.values[featureMatrixX][featureMatrixY] >= maxFeatureCorrelation or featureMatrix.values[featureMatrixX][featureMatrixY] <= -maxFeatureCorrelation:
                    newFeaturesWithoutY = np.delete(pickedFeatures, np.where(pickedFeatures == featureMatrixY))

                    pickedFeatures = newFeaturesWithoutY

            return pickedFeatures

    def _tryRemoveFeatures(self, clf, pickedFeaturesIndexes, f1DiffTreshold):
        bestNewMetric = self.currentMetrics
        bestNewPickedFeatures = pickedFeaturesIndexes;

        for featureIndexX in pickedFeaturesIndexes:
            newPickedFeatures = np.delete(pickedFeaturesIndexes, np.where(pickedFeaturesIndexes == featureIndexX))

            clfMetrics = self.testModel(clf, newPickedFeatures)

            if bestNewMetric.isBetter(clfMetrics, f1DiffTreshold) :
                bestNewMetric = clfMetrics
                bestNewPickedFeatures = newPickedFeatures
                self.currentMetrics = bestNewMetric

        return bestNewPickedFeatures

    def _tryRemoveUncorrelatedFeatures(self, clf, pickedFeatures, f1DiffTreshold):
        bestNewMetric = self.currentMetrics
        bestNewPickedFeatures = pickedFeatures;

        for featureIndexX in range(len(pickedFeatures[0])):
            newPickedFeatures = np.delete(pickedFeatures, featureIndexX) #np.delete(pickedFeaturesIndexes, np.where(pickedFeaturesIndexes == featureIndexX))

            clfMetrics = self.testModel(clf, newPickedFeatures)

            if bestNewMetric.isBetter(clfMetrics, f1DiffTreshold) :
                bestNewMetric = clfMetrics
                bestNewPickedFeatures = newPickedFeatures

        return bestNewPickedFeatures


    def testModel(self, clf, featuresIndexes):
        nSplits = 3
        sgkf = model_selection.StratifiedKFold(n_splits=nSplits, shuffle=False)

        avgAccuracy = []
        avgF1Score = []

        featuresDataset = np.take(self.featuresDataset, featuresIndexes, axis=1)

        for i in range(4):
            for trainIndexes, testIndexes in sgkf.split(featuresDataset, self.featuresDatasetLabels):
                kfold_train_dataset = featuresDataset[trainIndexes]
                kfold_train_dataset_labels = self.featuresDatasetLabels[trainIndexes]
                kfold_test_dataset =  featuresDataset[testIndexes]
                kfold_test_dataset_labels = self.featuresDatasetLabels[testIndexes]

                clf.fit(kfold_train_dataset, kfold_train_dataset_labels)
                kfold_predicted_labels = clf.predict(kfold_test_dataset)

                avgAccuracy.append(metrics.accuracy_score(kfold_test_dataset_labels, kfold_predicted_labels))
                avgF1Score.append(metrics.f1_score(kfold_test_dataset_labels, kfold_predicted_labels, average='weighted'))


        avgAccuracy = np.mean(avgAccuracy)
        avgF1Score = np.mean(avgF1Score)

        return ClfMetrics(avgAccuracy, avgF1Score)


    def calculateFeaturesCorrelation(self,pickedFeatures,algorithm='pearson'):
        reversedTrainDataset = dict()
        for trainArrIndex in range(len(pickedFeatures)):
            for featureTrainArrIndex in range(len(pickedFeatures[trainArrIndex])):
                colName = featureTrainArrIndex.__str__()
                if colName not in reversedTrainDataset: reversedTrainDataset[colName] = []
                reversedTrainDataset[colName].append(pickedFeatures[trainArrIndex][featureTrainArrIndex])

        xyz = pd.DataFrame(reversedTrainDataset)

        return xyz.corr(method=algorithm).round(decimals=2)


    def testingKeras(self,dataset_train_handcrafted_features,trainDatasetLabels) :
        inp = tf.convert_to_tensor(dataset_train_handcrafted_features)

        # Then, assuming X is a batch of input patterns:

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(5)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(inp, tf.convert_to_tensor(trainDatasetLabels), epochs=10,
                            validation_data=(inp, trainDatasetLabels))
