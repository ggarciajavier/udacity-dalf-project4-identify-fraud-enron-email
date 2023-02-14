#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from tester import dump_classifier_and_data

from helper_functions import AddRatio2Data, TestClassifier, AddScaledFeatures

### Task 1: Select what features you'll use.

## Features are selected based on the analysis shown in the documentation.
payments_features = ['salary', 'bonus', 'expenses']
stock_value_features =  ['total_stock_value']
email_features = ['from_poi_to_this_person', 'shared_receipt_with_poi']

features_list = ['poi'] + payments_features + stock_value_features + email_features

print('Features used for the analysis:')
print(features_list)
print('')

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s)
## Store to my_dataset for easy export below.
my_dataset = data_dict

scaled_features_dict = {'salary': 'salary_scaled',
                        'bonus': 'bonus_scaled',
                        'expenses': 'expenses_scaled',
                        'total_stock_value': 'total_stock_value_scaled',
                        'from_poi_to_this_person': 'from_poi_to_this_person_scaled',
                        'shared_receipt_with_poi': 'shared_receipt_with_poi_scaled'}

# Add scaled features to the dataset
my_dataset = AddScaledFeatures(my_dataset, scaled_features_dict)

## Add new features for percentage of salary, bonus and expenses vs. total payments
my_dataset = AddRatio2Data(my_dataset, 'salary', 'total_payments', 
                           'ratio_salary2total')
my_dataset = AddRatio2Data(my_dataset, 'bonus', 'total_payments', 
                           'ratio_bonus2total')
my_dataset = AddRatio2Data(my_dataset, 'expenses', 'total_payments', 
                           'ratio_expenses2total')

## Add new features for percentage of messages received, sent and shared with poi
my_dataset = AddRatio2Data(my_dataset, 'from_poi_to_this_person', 
                           'to_messages', 'ratio_received_from_poi')
my_dataset = AddRatio2Data(my_dataset, 'from_this_person_to_poi', 
                           'from_messages', 'ratio_sent_to_poi')
my_dataset = AddRatio2Data(my_dataset, 'shared_receipt_with_poi', 
                           'to_messages', 'ratio_shared_with_poi')


### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall

## Create a NumPy array containing the dataset
data = featureFormat(my_dataset, features_list, remove_NaN=True, 
                     remove_all_zeroes=False, remove_any_zeroes=False, 
                     sort_keys=True)

## Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
print('Testing GaussianNB classifier with the following parameters:')
print(clf_nb)
print('Results:')
print(TestClassifier(clf_nb, data, feature_scaling=False))
print('')

## K neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier()
print('Testing KNeighborsClassifier classifier with the following parameters:')
# The tuning of the classifier has been done in the documentation.
#  The following parameters showed the best performance.
parameters_knn = {'n_neighbors': 3, 'n_jobs': None, 'algorithm': 'auto', 
                  'metric': 'minkowski', 'metric_params': None, 'p': 2, 
                  'weights': 'distance', 'leaf_size': 30}
clf_knn.set_params(**parameters_knn)
print(clf_knn)
print('Results:')
print(TestClassifier(clf_knn, data, feature_scaling=False))
print('')

## SVM classifier
from sklearn.svm import SVC
clf_svc = SVC()
print('Testing SVM classifier with the following parameters:')
# The tuning of the classifier has been done in the documentation.
#  The following parameters showed the best performance.
parameters_svc = {'kernel': 'linear', 'C': 0.5, 'verbose': False, 
                  'probability': False, 'degree': 3, 'shrinking': True, 
                  'max_iter': -1, 'decision_function_shape': 'ovr', 
                  'random_state': None, 'tol': 0.001, 'cache_size': 1000, 
                  'coef0': 0.0, 'gamma': 'scale', 'class_weight': 'balanced'}
clf_svc.set_params(**parameters_svc)
print(clf_svc)
print('Results:')
print(TestClassifier(clf_svc, data, feature_scaling=True))
print('')

## Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier()
print('Testing Decision Tree classifier with the following parameters:')
# The tuning of the classifier has been done in the documentation.
#  The following parameters showed the best performance.
parameters_tree = {'presort': False, 'splitter': 'best', 
                   'min_impurity_decrease': 0.0, 'max_leaf_nodes': None, 
                   'min_samples_leaf': 12, 'min_samples_split': 2, 
                   'min_weight_fraction_leaf': 0.17, 'criterion': 'entropy', 
                   'random_state': None, 'min_impurity_split': None, 
                   'max_features': None, 'max_depth': 3, 
                   'class_weight': 'balanced'}
clf_tree.set_params(**parameters_tree)
print(clf_tree)
print('Results:')
print(TestClassifier(clf_tree, data, feature_scaling=False))
print('')


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

## Final classifier is the SVC
clf = clf_svc

## Since the SVC requires feature scaling, and the test_classifier function does not
## have a feature scaling option, I add the scaled features to the dataset
# Create a dictionary with the features to be scaled and the name of the scaled features
scaled_features_dict = {'salary': 'salary_scaled',
                        'bonus': 'bonus_scaled',
                        'expenses': 'expenses_scaled',
                        'total_stock_value': 'total_stock_value_scaled',
                        'from_poi_to_this_person': 'from_poi_to_this_person_scaled',
                        'shared_receipt_with_poi': 'shared_receipt_with_poi_scaled'}

# Add scaled features to the dataset
my_dataset = AddScaledFeatures(my_dataset, scaled_features_dict)
# Define new list of features with the scaled features
my_features_list = ['poi', 'salary_scaled', 'bonus_scaled', 'expenses_scaled',
                    'total_stock_value_scaled', 'from_poi_to_this_person_scaled',
                    'shared_receipt_with_poi_scaled']

dump_classifier_and_data(clf, my_dataset, my_features_list)