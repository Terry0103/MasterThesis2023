import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from ClassImp_Resampling.ClassImp import IHOT
from kee_utils.utils import import_dataset
from kee_utils.evaluation_pipeline import Train_model

logging.basicConfig(level = logging.INFO,
            format = "%(asctime)s %(levelname)s %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S",
            filename = './log/test_IHOT.log')


def test_IHOT():
    np.random.seed(30678)
    data = import_dataset('./Dataset/glass2.dat').select_dtypes(['number', 'category'])


    classifier = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 2)
    model = Train_model(train_data = data, resampler = None, classifier = classifier)
    model.evaluation()
    logging.info(f'Original evaluation outcome :\n{model.evaluation_outcome.iloc[0,:]}')

    model.set_params(**{'train_data': None, 'classifier': None, 'parameters_list': None, 'resampler': None})

    resampler = IHOT(n_neighbors = 3, kNN = None, classifier = None, optimization = 'saturation', max_saturation = 7)
    resampler.set_params(**{'classifier' : classifier})
    model = Train_model(train_data = data, resampler = resampler, classifier = classifier)
    model.evaluation()
    logging.info(f'Revised evaluation outcome :\n{model.evaluation_outcome.iloc[0,:]}')
    logging.info(resampler)

    logging.info(f'Test progress is over.')
    logging.info('-'*100)

def test_class_impurity():
    data = import_dataset('./Dataset/ecoli4.dat').select_dtypes(['number', 'category'])
    id = np.random.randint(0, data.shape[0], size = int(data.shape[0] * 0.75))
    X, Y = data.iloc[id, 0:-1].reset_index(drop = True), data.iloc[id, -1].reset_index(drop = True)
    print(X.shape)

    classifier = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 2)
    resampler = IHOT(n_neighbors = 3, kNN = None, classifier = None, optimization = 'best', max_saturation = 3)
    print(resampler)
    # print(dir(resampler))
    resampler.set_params(**{'classifier' : classifier})
    x, y = resampler.fit_resample(X, Y)
    # print(x.shape)
    print(resampler)

    
if __name__ == '__main__':
    test_IHOT()
    # test_class_impurity()
    # print(help(IHOT))