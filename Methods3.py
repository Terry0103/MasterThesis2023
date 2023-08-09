import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from ClassImp_Resampling.ClassImp import ClassImpurity, IHOT
from ClusterImp_Resampling.Cluster_impurity import ClusterImpurity
from kee_utils.utils import import_dataset

def main():
    # data = import_dataset('./Dataset/ecoli4.dat')
    # data = data.select_dtypes(['number', 'category'])
    # print(data['Class'].value_counts())
    # classifier = DecisionTreeClassifier
    # clustering = KMeans(n_init = 10)
    test = pd.DataFrame([[1, 2, 1], [1, 0, 1], [10, 4, 0], [10, 0, 0], [10, 2, 0], [1, 4, 1], [10, 4, 0], [10, 4, 0], [10, 4, 0], [10, 4, 0]])
    cls_imp = ClassImpurity()
    cls_imp.fit_class_impurity(X = test.iloc[:, 0:-1], Y = test.iloc[:, -1])
    # clu_imp = ClusterImpurity(n_clusters = 4, clustering_method = clustering).fit_cluster_impurity(X = data.iloc[:, 0:-1], Y = data.iloc[:, -1])
    # clu_imp = clu_imp.reshape([len(clu_imp), 1])

    # data['Cluster_imp'] = clu_imp
    # test = data.groupby(['Cluster_imp', 'Class']).count()
    # print(test)

    
if __name__ == '__main__':
    main()