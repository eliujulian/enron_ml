#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kendalltau
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import tester
import copy


print "# Starting #"

### Load the dictionary containing the dataset
print "# Loading Data #"
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Remove outliers
print "# Removing Outliers #"
data_dict.pop('TOTAL')

### Create new features

print "# Data Exploration #"


def get_nan_ratios(ratio=0.0):
    """
    Function finds numbers of NaN in a row.
    """
    result = []
    for x in data_dict['SKILLING JEFFREY K'].keys():
        r = float(len([v[x] for k, v in data_dict.iteritems() if v[x] == "NaN"])) / float(len(data_dict))
        if r >= ratio:
            result.append((x, r))
    return result


if False:
    # total number of data points: 145
    print "total number of data points:", len(data_dict)
    # allocation across classes (POI/non-POI): 18  /  127
    print "allocation across classes (POI/non-POI):", sum([v['poi'] for k, v in data_dict.iteritems()]), " / ", \
        len(data_dict) - sum([v['poi'] for k, v in data_dict.iteritems()])
    # features with more than 70 % NaNs [('restricted_stock_deferred', 0.8827586206896552),
    # ('loan_advances', 0.9793103448275862), ('director_fees', 0.8896551724137931)]
    print "features with more than 75 % NaNs", get_nan_ratios(ratio=0.75)


## Creating discrete features from continious features that are mostly 'NaN'.
## Choosing continious features with more than 80 % 'NaN's. Assigning 0 if NaN or == 0, else 1.


print "# Creating new features #"


def new_discrete_feature(feature):
    """
    Function to create new discrete features.
    """
    for k, v in data_dict.iteritems():
        if v[feature] == 'NaN' or v[feature] == 0:
            v[feature + "_discrete"] = 0
        else:
            v[feature + "_discrete"] = 1

# Here are the new features created
for n in get_nan_ratios(ratio=0.8):
    new_discrete_feature(n[0])


def analyze_email():
    "Helper funtion to look at the email feature."
    for k, v in data_dict.iteritems():
        if v['email_address'] == "NaN":
            pprint(k)
            pprint(data_dict[k])


## Creating a feature that counts the number von NaNs for each person

def count_nans_for_one_person(person_dict):
    result = 0
    for k, v in person_dict.iteritems():
        if v == 'NaN':
            result += 1
    return result


def count_all_nans(add_to_dict=False):
    result = {}
    for k, v in data_dict.iteritems():
        c = count_nans_for_one_person(v)
        result[k] = c
        if add_to_dict:
            v['count_nans'] = c
    return result

count_all_nans(add_to_dict=True)


## Feature counts words in name/key

def count_words_in_key(key):
    return len(key.split())


def count_all_words(add_to_dicts=False):
    result = {}
    for k, v in data_dict.iteritems():
        c = count_words_in_key(k)
        result[k] = c
        if add_to_dicts:
            v['count_words'] = c
    return result

count_all_words(add_to_dicts=True)


### Analyzing the features

# Helper to loop over the features
all_features = list(data_dict['SKILLING JEFFREY K'].keys())
all_features.pop(6) # Remove email


# I used the following features to select some feature to start feeding into the classifiers
def correlation_with_nan_as_zero():
    """
    Correlation between a feature an the label by pearsons r.
    """
    print "### Correaltion with NaN ###"
    labels = [v['poi'] for _, v in data_dict.iteritems()]

    def tmp():
        result = {}
        for f in all_features:
            x, y = [], []
            for a, b in zip(labels, [v[f] for _, v in data_dict.iteritems()]):
                if b != "NaN":
                    x.append(a)
                    y.append(b)
                else:
                    x.append(a)
                    y.append(0)
            result[f] = pearsonr(x, y)[0]
        return result

    return tmp()


def get_features_from_correlation(min_corr=0.26):
    """
    Function returns features based on correlation to the poi label.
    """
    features_list = ['poi']

    for k, v in correlation_with_nan_as_zero().iteritems():
        if abs(v) >= min_corr:
            # print k, v
            if k != 'poi':
                features_list.append(k)
    # print "--- ", len(features_list), "# features ---"
    return features_list


### Store to my_dataset for easy export#  below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing


# The following feature lists are the result of a add and substract search with Naive Bayes, Desicion Tree and
# K-Neighbors.
# started with: ['poi', 'deferred_income', 'count_nans', 'bonus', 'total_stock_value', 'salary',
# 'exercised_stock_options']

starting_features = ['poi', 'deferred_income', 'count_nans', 'bonus', 'total_stock_value', 'salary',
                     'exercised_stock_options']

best_nb_feature_list = ['poi', 'deferred_income', 'total_stock_value', 'salary', 'exercised_stock_options', 'expenses']

best_dt_feature_list = ['poi', 'deferred_income', 'exercised_stock_options', 'expenses', 'deferral_payments',
                        'director_fees']

best_knn_feature_list_scaled = ['poi', 'deferred_income', 'count_nans', 'total_stock_value',
                                'exercised_stock_options', 'restricted_stock_deferred',
                                'restricted_stock_deferred_discrete', 'deferral_payments']

best_knn_feature_list_not_scaled = ['poi', 'bonus', 'salary', 'count_nans', 'exercised_stock_options',
                                    'deferral_payments', 'other']

best_adaboost_feature_list = ['poi', 'deferred_income', 'count_nans', 'salary', 'exercised_stock_options',
                              'expenses', 'to_messages', 'shared_receipt_with_poi', 'other', 'from_poi_to_this_person']

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# The following functions are used for optimization. For local validation I used a simple loop over a train_test_split
# Besides this validation was mostly done by directly using the tester.main()

## Here are five models:

def NB_model():
    return GaussianNB()


def DT_model(max_depth=None):
    return DecisionTreeClassifier(max_depth=max_depth)


def KNN_model_not_scaled(n_neighbors=5, weights='distance'):
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)


def KNN_model_scaled(n_neighbors=5, weights='uniform'):
    return Pipeline([('scaling', preprocessing.StandardScaler()),
                     ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))])


def AdA_model():
    """
    Initial results were good but training and testing is a lot slower than the other models.
    """
    return AdaBoostClassifier()


# My own tester
def test_classifier(features, labels, clf):
    """
    The difference to the tester.py is that the metrics are calculated for every run an averaged at the end.
    """
    accu, r_precision, r_recall, r_f1 = [], [], [], []
    cv = StratifiedShuffleSplit(n_splits=2500, test_size=0.3, random_state=42)
    for train_indx, test_indx in cv.split(features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []

        for i in train_indx:
            features_train.append(features[i])
            labels_train.append(labels[i])
        for i in test_indx:
            features_test.append(features[i])
            labels_test.append(labels[i])

        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        if sum(pred) > 0:
            accu.append(clf.score(features_test, labels_test))
            r_precision.append(precision_score(labels_test, pred))
            r_recall.append(recall_score(labels_test, pred))
            r_f1.append(f1_score(labels_test, pred))

    if True:
        print "Classifier:", clf
        print "Accuracy:", np.mean(accu), "Precision:", np.mean(r_precision), \
            "Recall", np.mean(r_recall), "F1", np.mean(r_f1)
        print ""


# wrapper for the tester.py
def baseline_tester(features_list, classifier):
    dump_classifier_and_data(classifier, my_dataset, features_list)
    tester.main()


# wrapper for my own tester
def baseline_own_testing(feat, cls):
    labels, features = targetFeatureSplit(featureFormat(my_dataset, feat,
                                                        sort_keys=True))
    test_classifier(features, labels, cls)

# Run the tester
if False:
    print "# Running a Model #"
    # baseline_tester(best_knn_feature_list_not_scaled, KNN_model_not_scaled(weights="uniform"))
    baseline_own_testing(best_knn_feature_list_not_scaled, KNN_model_not_scaled(weights="uniform"))


# Function to add (and substract) features for search for best feature combination
def vary_features_add(features, classifier):
    print "Baseline - adding"
    baseline_tester(features, classifier)

    original = copy.copy(features)
    for feat in all_features:
        print "trying: ", feat
        features = copy.copy(original)

        if feat in features:
            continue
        else:
            features.append(feat)
        print "features used:", len(features), " -> ", features

        dump_classifier_and_data(classifier, my_dataset, features)
        tester.main()


def vary_features_eliminate(features, classifier):
    print "Baseline - Elimination"
    baseline_tester(features, classifier)

    original = copy.copy(features)
    for i, feat in enumerate(features[1:]):
        features = copy.copy(original)
        print "trying", feat
        features.pop(i + 1)
        print "features used:", len(features), " -> ", features
        dump_classifier_and_data(classifier, my_dataset, features)
        tester.main()


# wrapper for the vary functions
def vary_wrapper(feat_list, model):
    print "# Varying features #"
    print "## Elimiating ##"
    vary_features_eliminate(feat_list, model)
    print "## Adding ##"
    vary_features_add(feat_list, model)


# Run the wrapper
if False:
    vary_wrapper(starting_features, AdA_model())


# some tuning functions to find the best params for the models
def tuning_knn_scaled():
    """
    Best: k = 5, weights = 'uniform'
    """
    for n in range(1, 11):
        print n, 'uniform'
        baseline_tester(best_knn_feature_list_scaled, KNN_model_scaled(n, 'uniform'))
        print n, 'distance'
        baseline_tester(best_knn_feature_list_scaled, KNN_model_scaled(n, 'distance'))


def tuning_knn_not_scaled():
    """
    Best: k = 5, weights = 'distance'
    """
    for n in range(1, 11):
        print n, 'uniform'
        baseline_tester(best_knn_feature_list_not_scaled, KNN_model_not_scaled(n, 'uniform'))
        print n, 'distance'
        baseline_tester(best_knn_feature_list_not_scaled, KNN_model_not_scaled(n, 'distance'))


def tuning_dt():
    """
    Best: max_depth=None
    Most important features:
    clf.feature_importances_ -> [ 0.12174062  0.11639965  0.30168325  0.35446053  0.          0. 0.10571595]
    Most important: 'expenses', 2nd: 'exercised_stock_options'
    """
    if True:
        print "max depth"
        print "none"
        baseline_tester(best_dt_feature_list, DT_model(max_depth=None))
        for n in range(1, 10):
            print n
            baseline_tester(best_dt_feature_list, DT_model(max_depth=n))
    if True:
        labels, features = targetFeatureSplit(featureFormat(my_dataset, best_knn_feature_list_scaled,
                                                            sort_keys=True))
        clf = DT_model()
        clf.fit(features, labels)
        print "Feature Importances"
        print clf.feature_importances_


def show_final_scores():
    """
    Function shows all classifiers and their performance on the tester.py.
    """
    print ""
    print "*** Final Scores ***"
    print ""
    print "AdaBoost"
    dump_classifier_and_data(AdA_model(), my_dataset, best_adaboost_feature_list)
    tester.main()
    print ""
    print "Naive Bayes"
    dump_classifier_and_data(NB_model(), my_dataset, best_nb_feature_list)
    tester.main()
    print ""
    print "Desicion Tree"
    dump_classifier_and_data(DT_model(), my_dataset, best_dt_feature_list)
    tester.main()
    print ""
    print "K-Neigbohrs scaled"
    print "features used", best_knn_feature_list_scaled
    dump_classifier_and_data(KNN_model_scaled(), my_dataset, best_knn_feature_list_scaled)
    tester.main()
    print ""
    print "K-Neighbors not scaled"
    print "features used", best_knn_feature_list_not_scaled
    dump_classifier_and_data(KNN_model_not_scaled(), my_dataset, best_knn_feature_list_not_scaled)
    tester.main()


if True:
    show_final_scores()


if True:
    # Surprisingly I was not able to train the K-Neighbors algorithm for the scaled features as well as for the
    # unscaled. With a Precision of .73, a recall of .43 and a F1 of .54 this seems to be the best. I used
    # n_neighbors = 5 and weights = 'distance'
    print "# Submitting final model: K-Neighbors not scaled #"
    dump_classifier_and_data(KNN_model_not_scaled(),
                             my_dataset,
                             best_knn_feature_list_not_scaled)

print "# End #"
