from kee_utils.utils import *
from kee_utils.evaluation_pipeline import Train_model
from sklearn.tree import DecisionTreeClassifier
import smote_variants as sv

@timer
def utils():
    data = import_dataset('./Dataset/ecoli4.dat')
    a = read_data_list(path = './Dataset/', fileExt = '.dat')
    print(a)
    print(data.head())

def test_train_model():
    data = import_dataset('./Dataset/ecoli4.dat')
    classifier = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 2)
    resampler = sv.ROSE()
    model = Train_model(train_data = data, classifier = classifier, resampler = resampler)
    model.evaluation()
    print(model.evaluation_outcome)

if __name__ == '__main__':
    utils()
    test_train_model()