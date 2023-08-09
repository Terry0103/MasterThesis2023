import numpy as np
import pandas as pd
import logging
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from ClusterImp_Resampling.CIMPO import ClusIBasedOversampling
from kee_utils.utils import import_dataset

logging.basicConfig(level = logging.INFO,
            format = "%(asctime)s %(levelname)s %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S",
            filename = './log/visualization.log')


def visualization():
    np.random.seed(321)
    data = import_dataset('./Dataset/abalone19.dat')
    data = data.select_dtypes(['number', 'category'])

    pca = PCA(n_components = data.shape[1] - 1) # without target variable
    pca.fit(X = data.iloc[:, 0:-1])
    pca_data = pca.transform(X = data.iloc[:, 0:-1])[:, 0:2]



    pca_data = pd.DataFrame(pca_data, columns = ['COM1', 'COM2'])
    pca_data['Class'] = data.iloc[:, -1]
    sns.scatterplot(data = pca_data, x = 'COM1', y = 'COM2', hue = 'Class')
    plt.show()
    # for i in range(data.shape[1] - 1):
    #     for j in range(data.shape[1] - 1):
    #         if i == j:
    #             continue
    #         d1 = str(i)
    #         d2 = str(j)
    #         print(f'x: {d1}, y: {d2}')
    #         sns.scatterplot(data = data, x = d1, y = d2, hue = 'Class')
    #         plt.show()

    classifier = DecisionTreeClassifier(
        min_samples_split = 10,
        min_samples_leaf = 2)

    resampler = ClusIBasedOversampling(
        clustering_method = KMeans(n_init = 10),
        n_neighbors = 5,
        n_clusters = 4,
        early_stopping = 3,
        classifier = classifier,
        metric = balanced_accuracy_score)
    
    print(pca_data.shape)
    print(pca)
    print(f'PCA explained variance: {pca.explained_variance_ratio_}.')
    resampler.fit_resample(X = pca_data.iloc[:, 0:-1], Y = pca_data.iloc[:, -1])

    # logging.info(f'Test progress is over.')
    # logging.info('-'*100)

if __name__ == '__main__':
    visualization()