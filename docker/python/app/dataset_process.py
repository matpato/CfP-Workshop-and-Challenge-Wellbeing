import numpy as np


class FilteredDataset:
    def __init__(self, ecgFeatures, gsrFeatures,labels):
        self.ecgFeatures = ecgFeatures
        self.gsrFeatures = gsrFeatures
        self.labels = labels


class DatasetOperations:

    def createFilteredDataset(handcraftedFeaturesDataset, handcraftedFeaturesLabels):

        indexesToKeep = range(len(handcraftedFeaturesLabels))
        # else: indexesToKeep = np.delete(range(len(handcraftedFeaturesLabels)), np.unique(indexesToRemove))

        return FilteredDataset(ecgFeatures=handcraftedFeaturesDataset['ECG_features'][indexesToKeep],
                               gsrFeatures=handcraftedFeaturesDataset['GSR_features'][indexesToKeep],
                               labels=handcraftedFeaturesLabels[indexesToKeep])


    @staticmethod
    def filterZerosHandCraftedFeatures(handcraftedFeaturesDataset, handcraftedFeaturesLabels, filterZeros=True):

        # retornar array com indexes a manter ? (Masi facil)
        indexesToRemove = []

        for key in handcraftedFeaturesDataset.keys():
            if "masking" not in key: continue
            if not filterZeros: continue
            for i in range(len(handcraftedFeaturesDataset[key])):
                if np.min(handcraftedFeaturesDataset[key][i]) == 0:
                    indexesToRemove.append(i)

        indexesToKeep = range(len(handcraftedFeaturesLabels))
        # else: indexesToKeep = np.delete(range(len(handcraftedFeaturesLabels)), np.unique(indexesToRemove))

        return FilteredDataset(ecgFeatures=handcraftedFeaturesDataset['ECG_features'][indexesToKeep],
                               gsrFeatures=handcraftedFeaturesDataset['GSR_features'][indexesToKeep],
                               labels=handcraftedFeaturesLabels[indexesToKeep])

    @staticmethod
    def balancedSubsample(x, y, subsample_size=1.0):

        class_xs = []
        min_elems = None

        for yi in np.unique(y):
            elems = x[(y == yi)]
            class_xs.append((yi, elems))
            if min_elems == None or elems.shape[0] < min_elems:
                min_elems = elems.shape[0]

        use_elems = min_elems
        if subsample_size < 1:
            use_elems = int(min_elems * subsample_size)

        xs = []
        ys = []

        for ci, this_xs in class_xs:
            if len(this_xs) > use_elems:
                np.random.shuffle(this_xs)

            x_ = this_xs[:use_elems]
            y_ = np.empty(use_elems)
            y_.fill(ci)

            xs.append(x_)
            ys.append(y_)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        return xs, ys

    @staticmethod
    def binarizeLabels(labels):
        return np.where(labels == 0, 0, 1)