import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import recall_score, balanced_accuracy_score
from ClusterImp_Resampling.CIMPO import ClusIBasedOversampling
from kee_utils.utils import import_dataset
from kee_utils.evaluation_pipeline import Train_model



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(name)s: %(asctime)s: %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
filehandler = logging.FileHandler('./log/test_CIBO.log')
filehandler.setFormatter(formatter)

logger.addHandler(filehandler)


def test_CIBO():
    logger.info(f'{"Test progress starts":-^120}')
    # import dataset and declear classifier
    dataname = 'glass2.dat'
    data = import_dataset('./Dataset/' + dataname)
    data = data.select_dtypes(['number', 'category'])
    classifier = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 2)
    # classifier = MLPClassifier(max_iter = 1000, early_stopping = True, n_iter_no_change = 15,solver = 'adam', random_state = 1)

    # Data without preprocessing
    model = Train_model(train_data = data, resampler = None, classifier = classifier)
    model.evaluation()
    logger.info(f'Dataset: {dataname}')
    logger.info(f'Original evaluation outcome :\n{model.evaluation_outcome.iloc[0,:]}')
    
    # Initialization of evaluation pipeline
    # model.set_params(**{'train_data': None, 'classifier': None, 'parameters_list': None, 'resampler': None})

    # Data with my resampling method
    resampler = ClusIBasedOversampling(clustering_method = HDBSCAN(min_cluster_size = 2), n_neighbors = 5, n_clusters = 5, early_stopping = 10, learning_rate = 0.1)
    # resampler = ClusIBasedOversampling(clustering_method = KMeans(n_init = 10), n_neighbors = 5, n_clusters = 5, early_stopping = 10, learning_rate = 0.1)
    resampler.set_params(**{'classifier' : classifier})
    # print(resampler.fit_resample(X = data.iloc[:, 0:-1], Y = data.iloc[:, -1]))

    model = Train_model(train_data = data, resampler = resampler, classifier = classifier)
    model.evaluation()
    logger.info(f'Revised evaluation outcome :\n{model.evaluation_outcome.iloc[0,:].T}')
    logger.info(resampler)




    logger.info(f'{"Test progress over":-^120}')

if __name__ == '__main__':
    np.random.seed(321)
    test_CIBO()
    # test_loop_CIBO()
    # data = import_dataset('./Dataset/ecoli4.dat')
    # data = data.select_dtypes(['number', 'category'])
    # classifier = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 2)
    # resampler = ClusIBasedOversampling(clustering_method = KMeans(n_init = 10), n_neighbors = 5, n_clusters = 10, early_stopping = 5, learning_rate = 1)
    # resampler.set_params(**{'classifier' : classifier})
    # resampler.fit_resample(data.iloc[:, 0:-1], data.iloc[:, -1])
