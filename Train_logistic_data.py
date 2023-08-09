# General framework
import numpy as np
import pandas as pd
from kee_utils.evaluation_pipeline import Train_model
import logging
# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# Resamplers
import smote_variants as sv
from ClusterImp_Resampling.CIMPO import ClusIBasedOversampling
from ClassImp_Resampling.ClassImp import IHOT
# Clustering methods
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering, KMeans

clf_dict = {
    'DT' : DecisionTreeClassifier(criterion = 'entropy', random_state = 30678, min_samples_split = 10, min_samples_leaf = 2),
    'RF' : RandomForestClassifier(random_state = 30678, n_estimators = 200, min_samples_split = 10, min_samples_leaf = 2, n_jobs = -1),
    'MLP': MLPClassifier(random_state = 30678, learning_rate = 'adaptive', solver = 'adam', learning_rate_init = 0.001, early_stopping = True, activation = 'logistic'),
    'NB': GaussianNB(),
    'SVM' : SVC(kernel = 'rbf', C = 1, random_state = 30678),
    'KNN' : KNeighborsClassifier(n_neighbors = 5, n_jobs = -1),
}

resampler_dict = {
    'None' : None,
    'ROS' : sv.ROSE(),
    'SMOTE': sv.SMOTE(random_state = 777, k_neighbors = 5),
    'B-SMOTE' : sv.Borderline_SMOTE1(random_state = 777, k_neighbors = 5, m_neighbors = 5),
    'ADASYN' : sv.ADASYN(random_state = 777, n_neighbors = 5, d_th = 0.75, beta = 1),
    'SL-SMOTE' : sv.Safe_Level_SMOTE(random_state = 777, n_neighbors = 5),
    'MWMOTE' : sv.MWMOTE(random_state = 777, k1 = 5, k2 = 3, k3 = 5, M = 3, cf_th = 5, cmax = 2),
    'KM-SMOTE' : sv.kmeans_SMOTE(random_state = 777, n_neighbors = 5, n_clusters=10, iter = 2),
    'NRAS' : sv.NRAS(random_state = 777, n_neighbors = 5),
    'SOMO' : sv.SOMO(random_state = 777, n_grid = 10, sigma = 0.2, learning_rate = 0.5, n_iter = 100),
    'KM4' : ClusIBasedOversampling(clustering_method = KMeans(n_init = 10), n_clusters = 4),
    'KM5' : ClusIBasedOversampling(clustering_method = KMeans(n_init = 10), n_clusters = 5),
    'AHC4' : ClusIBasedOversampling(clustering_method = AgglomerativeClustering(), n_clusters = 4),
    'AHC5' : ClusIBasedOversampling(clustering_method = AgglomerativeClustering(), n_clusters = 5),
    'HDB' : ClusIBasedOversampling(clustering_method = HDBSCAN(min_cluster_size = 2)),
    'IHOT' : IHOT(n_neighbors = 3, optimization = 'saturation', max_saturation = 3)
}

def main():
    logging.basicConfig(level = logging.WARN,
                format = "%(asctime)s %(levelname)s %(message)s",
                datefmt = "%Y-%m-%d %H:%M:%S",
                filename = './log/20230614_test_logistic.log')

    data = pd.read_csv('C:/Users/KEE/Desktop/Logistic_Data/train.csv')
    Numerical = ['AWND', 'TMAX', 'SNWD', 'SNOW', 'PRCP', 'PLANE_AGE', 'AVG_MONTHLY_PASS_AIRLINE', 
                'AVG_MONTHLY_PASS_AIRPORT', 'AIRLINE_AIRPORT_FLIGHTS_MONTH', 'AIRLINE_FLIGHTS_MONTH',
                'AIRPORT_FLIGHTS_MONTH', 'NUMBER_OF_SEATS','LONGITUDE', 'LATITUDE', 'CONCURRENT_FLIGHTS']

    num_data = data[Numerical + ['DEP_DEL15']]
    del data

    root = 'C:/Users/KEE/Desktop/Logistic_Data/10000_ins_outcome/'
    repeat = ['rep_' + str(x) for x in range(1, 6)]
    # repeat = ['rep_1']


    C_based = ['KM4', 'KM5', 'AHC4', 'AHC5', 'HDB']
    other = ['None', 'ROS', 'SMOTE', 'B-SMOTE', 'ADASYN', 'SL-SMOTE', 'MWMOTE', 'KM-SMOTE', 'NRAS', 'SOMO']
    eval_metrics = ['Accuracy', 'AUC', 'F1-score', 'Recall', 'Precision', 'G-mean']
    final_columns = ['Classifier', 'Resampler'] + eval_metrics

    # test
    rnd_sample = np.random.randint(low = 0, high = num_data.shape[0] - 1, size = 1_000)
    train = num_data.iloc[rnd_sample, :].reset_index(drop = True)
    c = clf_dict['DT']
    r = resampler_dict['KM5']
    r.set_params(**{'classifier': c})
    
    model = Train_model(train_data = train, resampler = r, classifier = c)
    model.evaluation()
    logging.warn(f'Outcome:\n{model.evaluation_outcome.iloc[0,:]}')

    # for rep in range(len(repeat)):
    #     np.random.seed(rep + 10)
    #     logging.warn(f'Repeat {rep} starts.')
    #     logging.warn(f'Random seed: {rep}')
    #     rnd_sample = np.random.randint(low = 0, high = num_data.shape[0] - 1, size = 10_000)
    #     train = num_data.iloc[rnd_sample, :]
        
    #     for clf in ['DT', 'RF', 'NB', 'MLP', 'KNN', 'SVM']:
    #         outcome = np.empty((1, 6))
    #         classifier = clf_dict[clf]
    #         logging.warn(f'{clf} starts to process.')
    #         for pre in C_based + other:
    #             resampler = resampler_dict[pre]
    #             logging.warn(f'resampler : {pre} starts to process.')
                
    #             if 'classifier' in dir(resampler):
    #                 resampler.set_params(**{'classifier' : classifier})
                
    #             model = Train_model(train_data = train, classifier = classifier, resampler = resampler)
    #             model.evaluation()
    
    #             outcome = np.vstack((outcome, np.array(model.evaluation_outcome.iloc[0,:])))
                
    #             if isinstance(resampler, ClusIBasedOversampling):
    #                 resampler.set_params(**{'best_score' : 0, '__saturate_count__' : 0, 'best_balanced_data' : None})

    #             model.set_params(**{'train_data': None, 'classifier': None, 'parameters_list': None, 'resampler': None})

            
    #         outcome = np.hstack((np.array(C_based + other).reshape((len(C_based + other), 1)), np.round(outcome[1:, :], 4)))
    #         outcome = np.hstack((np.full((outcome.shape[0], 1), clf), outcome))
    #         outcome = pd.DataFrame(outcome, columns = final_columns)
    #         logging.warn(f'{classifier} have been done.')
    #         logging.warn(f'Evaluation outcome : \n {outcome}')
        
    #         outcome.to_csv(r'C:/Users/KEE/Desktop/Logistic_Data/10000_ins_outcome/' + repeat[rep] + '/' + clf + '.csv', index = False)
    #         logging.warn(f"File is stored properly. File path : {'C:/Users/KEE/Desktop/Logistic_Data/10000_ins_outcome/' + repeat[rep] + '/' + clf + '.csv'}")
    #         logging.warn('-'*100)
            
    # logging.warn(f'Process END.')

if __name__ == '__main__':
    main()
