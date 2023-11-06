# General packages
import pandas as pd
import numpy as np
import time
import logging
# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# Resamplers
import smote_variants as sv
# Clustering methods
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
# self-defined modules
import kee_utils.evaluation_pipeline as F
from kee_utils.utils import *
from ClusterImp_Resampling.CIMPO import ClusIBasedOversampling


clf_dict = {
    'DT' : DecisionTreeClassifier(criterion = 'entropy', random_state = 30678, min_samples_split = 10, min_samples_leaf = 2),
    'RF' : RandomForestClassifier(random_state = 30678, n_estimators = 200, min_samples_split = 10, min_samples_leaf = 4, n_jobs = -1),
    'MLP': MLPClassifier(random_state = 30678, learning_rate = 'adaptive', solver = 'adam', learning_rate_init = 0.001, early_stopping = True, activation = 'logistic'),
    'MLP': MLPClassifier(max_iter = 1000, early_stopping = True, n_iter_no_change = 15,solver = 'adam', random_state = 1, activation = 'relu'),
    'NB': GaussianNB(),
    'SVM' : SVC(kernel = 'rbf', C = 1, random_state = 30678),
    'KNN' : KNeighborsClassifier(n_neighbors = 3, n_jobs = -1),
    None : None
}
parameter_dict = {
    'DT' :{'max_depth' : (1, 2, 4, 8), 'min_samples_split' : (2, 4, 8), 'min_samples_leaf' :(1, 2, 4, 8)},
    'RF' :{'max_depth' : (1, 2, 4, 8), 'min_samples_split' : (2, 4, 8), 'min_samples_leaf' :(1, 2, 4, 8)},
    'MLP':{'learning_rate_init' : (0.001, 0.01, 0.1)},
    'NB': {'var_smoothing' : (0.000000001, 0.0000001, 0.0001)},
    'SVM' : {'C' : (2, 4, 8, 16, 32), 'kernel' : ('linear', 'rbf'),},
    'KNN' : {'n_neighbors' : (2, 3, 4, 5)}
}
resampler_dict = {
    'None' : None,
    'ROS' : sv.ROSE(random_state = 777),
    'SMOTE': sv.SMOTE(random_state = 777, k_neighbors = 5),
    'B-SMOTE' : sv.Borderline_SMOTE1(random_state = 777, k_neighbors = 5, m_neighbors = 5),
    'ADASYN' : sv.ADASYN(random_state = 777, n_neighbors = 5, d_th = 0.75, beta = 1),
    'SL_SMOTE' : sv.Safe_Level_SMOTE(random_state = 777, n_neighbors = 5),
    'MWMOTE' : sv.MWMOTE(random_state = 777, k1 = 5, k2 = 3, k3 = 5, M = 3, cf_th = 5, cmax = 2),
    'KMeansSMOTE' : sv.kmeans_SMOTE(random_state = 777, n_neighbors = 5, n_clusters=10, iter = 2),
    'NRAS' : sv.NRAS(random_state = 777, n_neighbors = 5),
    'SOMO' : sv.SOMO(random_state = 777, n_grid = 10, sigma = 0.2, learning_rate = 0.5, n_iter = 100),
    # 'IHOT': IHOT(n_neighbors = 3, optimization = 'best')
}


def single_test(classifier, resampling_method, classifier_para_dict = None, store_model : bool = False, file_name : str = None):

    eval_metrics = ['Accuracy', 'AUC', 'F1-score', 'Recall', 'Precision', 'G-mean']
    outcome = pd.DataFrame(np.zeros((len(data_list), 6)), columns = eval_metrics, index = data_list)
    
    clf = clf_dict[classifier]
    resampler = resampler_dict[resampling_method]
    # final_shape = pd.DataFrame(np.zeros((len(data_list), 2)),dtype = float, index = data_list, columns = ['#Min', '#Maj'])

    ###################
    if 'classifier' in dir(resampler):
        resampler.set_params(**{'classifier' : clf})
    ###################
    logger.info(f'{classifier} starts to process.')
    time_stamp = np.full([len(data_list), 1], fill_value = 'None')
    
    for i in range(len(data_list)):
        timer = time.time()
        dataset = import_dataset(path = './Dataset/' + data_list[i] + '.dat')
        dataset = dataset.select_dtypes(['number', 'category'])
        
        # print(dataset.shape)
        # model = F.Train_model(train_data = dataset, classifier = clf, parameters_dict = para_dict, resampler = resampler)
        if 'classifier' in dir(resampler):
            resampler.set_params(**{'classifier' : clf})

        model = F.Train_model(train_data = dataset, classifier = clf, resampler = resampler)
        model.evaluation()
        outcome.iloc[i, :] = model.evaluation_outcome.iloc[0,:]
        
        time_stamp[i, 0] = np.round(time.time() - timer, 6)


        # if isinstance(resampler, ClusIBasedOversampling):
        #     resampler.set_params(**{'__best_score' : 0, '__saturate_count': 0})
        
        model.set_params(**{'train_data': None, 'classifier': None, 'resampler': None,})
        # final_shape = pd.concat((final_shape, model.final_shape()), axis = 1)
        # final_shape.iloc[i, 0] = min(model.final_shape())
        # final_shape.iloc[i, 1] = max(model.final_shape())
        
        logger.info(f'{classifier} : {data_list[i]} has compeleted!')

    outcome['ExeTime'] = time_stamp
    logger.info(f'Evaluation outcome : \n {outcome}')
    logger.info(f'{classifier} have been done.')
    logger.info('-'*100)
    if store_model:
            outcome.to_csv(r'./Experimental_result/' + file_name + '.csv', index = True)
        # final_shape.to_csv(r'C:/Users/KEE/Desktop/PF_SMOTE_DATASETS/EVL_outcome/' + name + '_final_shape.csv')



if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(name)s: %(asctime)s: %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    filehandler = logging.FileHandler('./log/2023110607_training.log')
    filehandler.setFormatter(formatter)

    logger.addHandler(filehandler)
    # import dataset
    data_list = read_data_list(path = './Dataset/')[3:7]
    start = time.time()

    cluster_dict = {
        'KM5' :  ClusIBasedOversampling(n_clusters = 5, clustering_method = KMeans(n_init = 10), early_stopping = 10, learning_rate = 0.1),
        'KM4' :  ClusIBasedOversampling(n_clusters = 4, clustering_method = KMeans(n_init = 10), early_stopping = 10, learning_rate = 0.1),
        'AHC5' : ClusIBasedOversampling(n_clusters = 5, clustering_method = AgglomerativeClustering(), early_stopping = 10, learning_rate = 0.1),
        'AHC4' : ClusIBasedOversampling(n_clusters = 4, clustering_method = AgglomerativeClustering(), early_stopping = 10, learning_rate = 0.1),
        # 'HDB_rm_noise' : ClusIBasedOversampling(clustering_method = HDBSCAN(min_cluster_size = 2), , early_stopping = 5),
        'HDB' : ClusIBasedOversampling(clustering_method = HDBSCAN(min_cluster_size = 2)),
    } 
    
    pre = 'CIBO'
    for clus in cluster_dict.keys():
        resampler_dict['CIBO'] = cluster_dict[clus]
        # for clf in ['DT', 'RF', 'MLP', 'NB', 'KNN', 'SVM']:
        # stucked ['MLP','SVM']:
        # good : [ 'DT','RF','NB','MLP']
        for clf in ['DT', 'RF', 'MLP', 'NB', 'KNN', 'SVM']:
            single_test(classifier = clf, resampling_method = pre, classifier_para_dict = None,
                        # store_model = False, file_name = 'MLP/' + pre + '_' +  clf)
                        store_model = False, file_name = '20231106_test/' + clus + '/' + pre + '_' +  clf)
                        # store_model = True, file_name = 'SVM_test_large_satu_limit_0605/' + 'linear' + clus + '_' + pre + '_' +  clf)

    ###########################################################################
    ########################################################################### 
    # for method_comparison 
    # classifer_list = ['DT', 'RF', 'MLP', 'NB', 'KNN', 'SVM']
    # method_Comparison()
    
    print('#'*40 + ' Process has completed ' + '#'*40)
    print('Execution time : %4d mins %.4f s' %((time.time() - start) // 60, (time.time()- start) % 60))