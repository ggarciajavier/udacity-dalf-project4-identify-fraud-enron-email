# -*- coding: utf-8 -*-
"""
A set of functions to support the processing of the data and the testing
and tuning of the machine learning classifiers used in the poi_id.py code.
The following functions are included:
    - AddRatio2Data
    - AddScaledFeatures
    - TuneClassifier
    - TestClassifier
    - RunClassifier
    - ScaleData
"""

from time import time
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from feature_format import targetFeatureSplit

def AddRatio2Data(data_dict, num_feature, den_feature, new_feature):
    """ Creates a new feature computed as a ratio between two other features
    present in the data. The function requires the following input variables:
        - data_dict: dictionary containing the data to be modified
        - num_feature: string with name of feature for numerator
        - den_feature: string with name of feature for denominator
        - new_feature: string with name of new feature
    """
    for key in data_dict.keys():
        num = float(data_dict[key][num_feature])
        den = float(data_dict[key][den_feature])
        if (math.isnan(num)) or (math.isnan(den)) or (den == 0.0):
            data_dict[key][new_feature] = 'NaN'
        else:
            data_dict[key][new_feature] = str(num / den)
    
    return data_dict


def AddScaledFeatures(data_dict, features_dict):
    """ Function to scale features directly in the dictionary containing the
    dataset. It is needed for including the scaled features in the output
    dataset for the .pkl files. The scaling is based on the MinMaxScaler 
    function of sklearn.
    The following inputs are required:
        - data_dict: dictionary containing the data to be modified
        - features_dict: dictionary containing the features to be scaled as keys
                         and the names of the scaled features
    """
    
    for original_feature in features_dict.keys():
        scaled_feature = features_dict[original_feature]
        feature_values = [0.0 if math.isnan(float(data_dict[key][original_feature]))
                          else float(data_dict[key][original_feature])
                          for key in data_dict.keys()]
        min_value = min(feature_values)
        max_value = max(feature_values)
        for key in data_dict.keys():
            original_value = float(data_dict[key][original_feature])
            if math.isnan(original_value):
                data_dict[key][scaled_feature] = 'NaN'
            else:
                data_dict[key][scaled_feature] = str((original_value - min_value)
                                                     / (max_value - min_value))
    
    return data_dict


def TuneClassifier(clf, parameter_dict, data, feature_scaling=False, folds=1000, test_p=0.1):
    """ Function to try different combinations of parameter values on a given
    classifier and dataset. It provides the result in a Pandas DataFrame for
    further analysis. 
    The following inputs are required:
        - clf: machine learning classifier (no parameters given)
        - parameter: dictionary with parameter names as keys and lists of 
                     parameter settings to try as values
        - data: NumPy array containing the input data (labels and features)
        - feature_scaling: boolean indicating whether features are to be scaled or not
        - folds: integer indicating the number of repetitions for the 
                 StratifiedShuffleSplit
        - test_p: float indicating the proportion of the dataset to include in
                  the test split done in StratifiedShuffleSplit
    """
    # Separate data in labels and features
    labels, features = targetFeatureSplit(data)
    # Scale features (if required)
    if feature_scaling:
        print('Feature scaling is active.')
        features = ScaleData(features)
    # Convert parameter dictionary into grid
    parameter_grid = ParameterGrid(parameter_dict)
    print('Testing a total of %i parameter combinations.' % (len(parameter_grid)))
    results_dict = defaultdict(list)
    # Loop parameter grid and run classifier
    for param_set in parameter_grid:
        # Set new parameters for classifier
        clf.set_params(**param_set)
        # Run classifier (training and evaluating) and store outputs in dictionary
        output_dict = RunClassifier(clf, labels, features, folds)
        # Append outputs to defaultdict
        results_dict['estimator'].append(clf.__class__.__name__)
        results_dict['parameters'].append(clf.get_params())
        for key in output_dict:
            results_dict[key].append(output_dict[key])
    
    # Store results in a DataFrame
    results_df = pd.DataFrame(data=results_dict)
    print("Classifier tuning finished.")
    
    return results_df


def TestClassifier(clf, data, feature_scaling=False, folds=1000, test_p=0.1):
    """ Function to test a classifier on a given dataset using the additional
    function RunClassifier. It provides the results in a dictionary. 
    The following inputs are required:
        - clf: machine learning classifier (with or without defined parameters)
        - data: NumPy array containing the input data (labels and features)
        - feature_scaling: boolean indicating whether features are to be scaled or not
        - folds: integer indicating the number of repetitions for the 
                 StratifiedShuffleSplit
        - test_p: float indicating the proportion of the dataset to include in
                  the test split done in StratifiedShuffleSplit            
    """
    # Separate data in labels and features
    labels, features = targetFeatureSplit(data)
    
    # Scale features
    if feature_scaling:
        print('Feature scaling is active.')
        features = ScaleData(features)
    t0 = time()
    results_dict = RunClassifier(clf, labels, features, folds, test_p)
    t1 = time()
    print('Elapsed time = %.2f seconds.' % round(t1 - t0, 2))
    return results_dict


def RunClassifier(clf, labels, features, folds=1000, test_p=0.1):
    """ Function to train and evaluate a classifier on a set of features and 
    labels. It uses StratifiedShuffleSplit for cross validation, training the
    classifier on the training dataset and evaluating it on the test set. It
    computes the average of the accuracy, precision, recall, f1 and f2 and
    stores them in a dictionary. The code is based on the test_classifier
    function provided in the tester.py. 
    The following inputs are required:
        - clf: machine learning classifier (with or without defined parameters)
        - labels: NumPy array with the input labels
        - features: NumPy array with the input features (scaled or not)
        - folds: integer indicating the number of repetitions for the 
                 StratifiedShuffleSplit
        - test_p: float indicating the proportion of the dataset to include in
                  the test split done in StratifiedShuffleSplit
    """
    
    cv = StratifiedShuffleSplit(n_splits=folds, test_size=test_p, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    results_dict = {}
    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = (true_negatives + false_negatives + false_positives 
                             + true_positives)
        results_dict['accuracy'] = round(1.0 * (true_positives + true_negatives) / 
                                         total_predictions, 5)
        results_dict['precision'] = round(1.0 * true_positives / 
                                          (true_positives + false_positives), 5)
        results_dict['recall'] = round(1.0 * true_positives / 
                                       (true_positives+false_negatives), 5)
        results_dict['f1'] = round(2.0 * true_positives / (2 * true_positives +
                                   false_positives + false_negatives), 5)
        results_dict['f2'] = round((1 + 2.0 * 2.0) * results_dict['precision'] 
                                   * results_dict['recall'] / (4 * 
                                   results_dict['precision'] + 
                                   results_dict['recall']), 5)
    except:
        print "Got a divide by zero when trying out:", clf
        print "All evaluation metrics set to NaN."
        results_dict['accuracy'] = np.nan
        results_dict['precision'] = np.nan
        results_dict['recall'] = np.nan
        results_dict['f1'] = np.nan
        results_dict['f2'] = np.nan
    
    return results_dict


def ScaleData(data_in):
    """ Scales data using the MinMaxScaler function from sklearn. 
    The following inputs are required:
        - data_dict: dictionary containing the data to be modified
    """
    
    min_max_scaler = MinMaxScaler()
    data_out = min_max_scaler.fit_transform(data_in)
    
    return data_out