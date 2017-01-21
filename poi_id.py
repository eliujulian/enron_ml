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
from sklearn.svm import SVC
import tester
import copy


random_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 29, 30, 21, 31, 42, 100]

print "Starting"

### Load the dictionary containing the dataset
print "Loading Data"
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Remove outliers
print "Removing Outliers"
data_dict.pop('TOTAL')

### Create new features


print "Creating new features"

## Creating discrete features from continious features that are mostly 'NaN'.
## Choosing continious features with more than 80 % 'NaN's. Assigning 0 if NaN or == 0, else 1.

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


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing


# The following feature lists are the result of a add and substract search with Naive Bayes, Desicion Tree and
# K-Neighbors.

best_nb_feature_list = ['poi', 'deferred_income', 'total_stock_value', 'salary', 'exercised_stock_options', 'expenses']

best_dt_feature_list = ['poi', 'deferred_income', 'exercised_stock_options', 'expenses', 'deferral_payments',
                        'director_fees']

best_knn_feature_list = ['poi', 'bonus', 'salary', 'count_nans', 'exercised_stock_options', 'deferral_payments',
                         'other']

features_list = best_knn_feature_list

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# The following functions are used for optimization. For local validation I looped 20 times over a train_test_split
# Besides this validation was mostly done by dumping the classifier, dataset and featurelist, importing the tester.py
# and running it.

def multi_tester_NB(random_states, features, labels):
    result = []
    for n in random_states:
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=n)
        clf = GaussianNB()
        clf.fit(features_train, labels_train)
        result.append(clf.score(features_test, labels_test))
    if True:
        print "Classifier:", "GaussianNB", " Score ave.", np.mean(result)
        print ""
    return clf.fit(features, labels)


def multi_tester_DT(random_states, features, labels):
    result = []
    for n in random_states:
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=n)
        clf = DecisionTreeClassifier()
        clf.fit(features_train, labels_train)
        result.append(clf.score(features_test, labels_test))
    if True:
        print "Classifier:", "Dessision Tree", len(random_states), "# runs, Score ave.",  np.mean(result)
        print ""

    return clf.fit(features, labels)


def multi_tester_KNN(random_states, features, labels):
    result = []
    for n in random_states:
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=n)
        clf = KNeighborsClassifier()
        clf.fit(features_train, labels_train)
        result.append(clf.score(features_test, labels_test))
    if True:
        print "Classifier:", "Dessision Tree", len(random_states), "# runs, Score ave.",  np.mean(result)
        print ""

    return KNeighborsClassifier(n_neighbors=5).fit(features, labels)


function_used = multi_tester_KNN

if True:
    # Baseline
    clf = function_used(random_states, features, labels)
    dump_classifier_and_data(clf, my_dataset, features_list)
    tester.main()


def vary_features(base_features, tester_function):
    # Function tries to add all features one by one and prints the scores. Elimination was done manually. Adding and
    # elimination were done in sequence until I didn't find significant improvements
    print "Baseline"
    data = featureFormat(my_dataset, base_features, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf = tester_function(random_states, features, labels)
    dump_classifier_and_data(clf, my_dataset, base_features)
    tester.main()

    original = copy.copy(base_features)
    for feat in all_features:
        print "trying: ", feat
        base_features = copy.copy(original)

        if feat in base_features:
            print "allready used"
            print ""
            continue
        else:
            base_features.append(feat)
        print "features used:", len(base_features), " -> ", base_features

        data = featureFormat(my_dataset, base_features, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        clf = tester_function(random_states, features, labels)
        dump_classifier_and_data(clf, my_dataset, base_features)
        tester.main()

if False:
    vary_features(base_features=features_list, tester_function=function_used)


def show_final_scores():
    print ""
    print "*** Final Scores ***"
    print ""
    print "Naive Bayes"
    data = featureFormat(my_dataset, best_nb_feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf = GaussianNB()
    clf.fit(features, labels)
    dump_classifier_and_data(clf, my_dataset, best_nb_feature_list)
    tester.main()
    print ""
    print "Desicion Tree"
    data = featureFormat(my_dataset, best_dt_feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf = DecisionTreeClassifier()
    clf.fit(features, labels)
    dump_classifier_and_data(clf, my_dataset, best_dt_feature_list)
    tester.main()
    print ""
    print "K-Neigbohrs"
    data = featureFormat(my_dataset, best_knn_feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(preprocessing.scale(features), labels)
    dump_classifier_and_data(clf, my_dataset, best_knn_feature_list)
    tester.main()


# show_final_scores()


if True:
    print "Submitting final model: K-Neighbors"
    labels, features = targetFeatureSplit(featureFormat(my_dataset, best_knn_feature_list, sort_keys=True))
    dump_classifier_and_data(KNeighborsClassifier(n_neighbors=5).fit(features, labels),
                             my_dataset,
                             best_knn_feature_list)

print "End"
