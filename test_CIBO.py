import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, balanced_accuracy_score
from ClusterImp_Resampling.CIMPO import ClusIBasedOversampling
from kee_utils.utils import import_dataset
from kee_utils.evaluation_pipeline import Train_model

logging.basicConfig(level = logging.INFO,
            format = "%(asctime)s %(levelname)s %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S",
            filename = './log/test_CIBO.log')


def test_CIBO():
    np.random.seed(321)
    data = import_dataset('./Dataset/ecoli4.dat')
    data = data.select_dtypes(['number', 'category'])
    classifier = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 2)
    # classifier = MLPClassifier(max_iter = 1000, early_stopping = True, n_iter_no_change = 15,solver = 'adam', random_state = 1)
    model = Train_model(train_data = data, resampler = None, classifier = classifier)
    model.evaluation()
    logging.info(f'Original evaluation outcome :\n{model.evaluation_outcome.iloc[0,:]}')

    model.set_params(**{'train_data': None, 'classifier': None, 'parameters_list': None, 'resampler': None})

    resampler = ClusIBasedOversampling(clustering_method = KMeans(n_init = 10), n_neighbors = 5, n_clusters = 5, early_stopping = 20, learning_rate = 0.05)
    resampler.set_params(**{'classifier' : classifier})
    model = Train_model(train_data = data, resampler = resampler, classifier = classifier)
    model.evaluation()
    logging.info(f'Revised evaluation outcome :\n{model.evaluation_outcome.iloc[0,:].T}')
    logging.info(resampler)

    logging.info(f'Test progress is over.')
    logging.info('-'*100)

    # print(model.resampler.__dir__())

def test_loop_CIBO():
    # logging.info('-'*50, 'test_CIBO_LOOP', '-'*50)
    np.random.seed(30678)
    data1 = import_dataset('./Dataset/glass2.dat').select_dtypes(['number', 'category'])
    data2 = import_dataset('./Dataset/ecoli4.dat').select_dtypes(['number', 'category'])
    data3 = import_dataset('./Dataset/zoo-3.dat').select_dtypes(['number', 'category'])

    classifier = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 2)
    resampler = ClusIBasedOversampling(clustering_method = KMeans(n_init = 10), n_neighbors = 5, n_clusters = 4, early_stopping = 3)


    for _ in [data1, data2, data3]:
        data = _
        # print(f'before shape: {data.shape}')

        if 'classifier' in dir(resampler):
            resampler.set_params(**{'classifier' : classifier})

        model = Train_model(train_data = data, resampler = resampler, classifier = classifier)
        model.evaluation()

        if isinstance(resampler, ClusIBasedOversampling):
            resampler.set_params(**{'best_score' : 0, 'saturate_count': 0})

        print(f'after shape: {resampler.best_balanced_data[0].shape}')
        # print('-'*100)

    # logging.info(f'Revised evaluation outcome :\n{model.evaluation_outcome.iloc[0,:]}')
    # logging.info(resampler)

    # logging.info(f'Test progress is over.')
    # logging.info('-'*100)

if __name__ == '__main__':
    test_CIBO()
    # test_loop_CIBO()
