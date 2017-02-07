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
from sklearn.model_selection import GridSearchCV
import tester
import copy


print "# Starting #"

# Load the dictionary containing the dataset
print "# Loading Data #"
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Remove outliers
print "# Removing Outliers #"
data_dict.pop('TOTAL')

# Create new features

print "# Data Exploration #"


def get_nan_ratios(ratio=0.0):
    """
    Function finds numbers of NaN in a row.
    """
    result = []
    for x in data_dict['SKILLING JEFFREY K'].keys():
        r = float(len([v_[x] for k_, v_ in data_dict.iteritems() if v_[x] == "NaN"])) / float(len(data_dict))
        if r >= ratio:
            result.append((x, r))
    return result


if False:
    print "total number of data points:", len(data_dict)
    # > total number of data points: 145

    print "allocation across classes (POI/non-POI):", sum([v['poi'] for k, v in data_dict.iteritems()]), " / ", \
        len(data_dict) - sum([v['poi'] for k, v in data_dict.iteritems()])
    # > allocation across classes (POI/non-POI): 18  /  127

    print "number of features in dataset:", len(list(data_dict['SKILLING JEFFREY K'].keys()))
    # > number of features in dataset: 21

    print "features with more than 75 % NaNs", get_nan_ratios(ratio=0.75)
    # > features with more than 75 % NaNs [('restricted_stock_deferred', 0.8827586206896552),
    # > ('loan_advances', 0.9793103448275862), ('director_fees', 0.8896551724137931)]


# Creating discrete features from continious features that are mostly 'NaN'.
# Choosing continious features with more than 80 % 'NaN's. Assigning 0 if NaN or == 0, else 1.


print "# Creating new features #"


def new_discrete_feature(feature):
    """
    Function to create new discrete features.
    """
    for k_, v_ in data_dict.iteritems():
        if v_[feature] == 'NaN' or v_[feature] == 0:
            v_[feature + "_discrete"] = 0
        else:
            v_[feature + "_discrete"] = 1

# Here are the new features created
for n in get_nan_ratios(ratio=0.8):
    new_discrete_feature(n[0])


def analyze_email():
    """
    Helper function to look at the email feature.
    """
    for k_, v_ in data_dict.iteritems():
        if v_['email_address'] == "NaN":
            pprint(k_)
            pprint(data_dict[k_])


# Creating a feature that counts the number von NaNs for each person

def count_nans_for_one_person(person_dict):
    result = 0
    for k_, v_ in person_dict.iteritems():
        if v_ == 'NaN':
            result += 1
    return result


def count_all_nans(add_to_dict=False):
    result = {}
    for k_, v_ in data_dict.iteritems():
        c = count_nans_for_one_person(v_)
        result[k_] = c
        if add_to_dict:
            v_['count_nans'] = c
    return result

count_all_nans(add_to_dict=True)


# Feature counts words in name/key

def count_words_in_key(key):
    return len(key.split())


def count_all_words(add_to_dicts=False):
    result = {}
    for k_, v_ in data_dict.iteritems():
        c = count_words_in_key(k_)
        result[k_] = c
        if add_to_dicts:
            v_['count_words'] = c
    return result

count_all_words(add_to_dicts=True)


# Analyzing the features

# Helper to loop over the features
all_features = list(data_dict['SKILLING JEFFREY K'].keys())
all_features.pop(6)  # Remove email


# I used the following features to select some feature to start feeding into the classifiers
def correlation_with_nan_as_zero():
    """
    Correlation between a feature an the label by pearsons r.
    """
    print "### Correaltion with NaN ###"
    labels = [v_['poi'] for _, v_ in data_dict.iteritems()]

    def tmp():
        result = {}
        for f in all_features:
            x, y = [], []
            for a, b in zip(labels, [v_1[f] for _, v_1 in data_dict.iteritems()]):
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


def get_pandas_dataframe(nan_as_zero=False):
    df = pd.DataFrame.from_dict(data_dict).transpose()
    if nan_as_zero:
        def convert(x):
            if isinstance(x, int):
                return x
            elif isinstance(x, str):
                if x == "NaN":
                    return 0
                elif x.isalnum():
                    return int(x)
                else:
                    raise Exception("Not possible")
        for c in all_features:
            df[c] = [convert(n_) for n_ in df[c]]
    return df


def get_starting_features_from_df(min_corr=0.26, print_results=False):
    df = get_pandas_dataframe(nan_as_zero=True).corr()
    df_corr_poi = df['poi']
    result = df_corr_poi[abs(df_corr_poi) >= min_corr]

    if print_results:
        print result.sort_values(ascending=False)

    result = [n_ for n_ in result.axes[0]]
    result.remove('poi')
    result.insert(0, 'poi')
    return result


# Store to my_dataset for easy export#  below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing


# The following feature lists are the result of a add and substract search with Naive Bayes, Desicion Tree and
# K-Neighbors.
# started with: ['poi', 'deferred_income', 'count_nans', 'bonus', 'total_stock_value', 'salary',
# 'exercised_stock_options']

starting_features = get_starting_features_from_df(print_results=False)

temp_list = ['poi', 'exercised_stock_options', 'deferral_payments']

best_nb_feature_list = ['poi',
                        'deferred_income',
                        'total_stock_value',
                        'salary',
                        'exercised_stock_options',
                        'expenses']

best_dt_feature_list = ['poi',
                        'deferred_income',
                        'exercised_stock_options',
                        'expenses',
                        'deferral_payments',
                        'director_fees']

best_knn_feature_list_scaled = ['poi',
                                'bonus',
                                'exercised_stock_options',
                                'total_stock_value',
                                'deferral_payments',
                                'restricted_stock_deferred_discrete',
                                'restricted_stock_deferred']

best_knn_feature_list_not_scaled = ['poi',
                                    'bonus',
                                    'exercised_stock_options',
                                    'salary',
                                    'loan_advances_discrete']

best_adaboost_feature_list = ['poi',
                              'deferred_income',
                              'count_nans',
                              'salary',
                              'exercised_stock_options',
                              'expenses',
                              'to_messages',
                              'shared_receipt_with_poi',
                              'other',
                              'from_poi_to_this_person']

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# The following functions are used for optimization. For local validation I used a simple loop over a train_test_split
# Besides this validation was mostly done by directly using the tester.main()

# Here are five models:


def NB_model():
    return GaussianNB()


def DT_model(max_depth=None):
    return DecisionTreeClassifier(max_depth=max_depth)


def KNN_model_not_scaled(n_neighbors=5, weights='distance'):
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)


def KNN_model_scaled(n_neighbors=3, weights='uniform'):
    return Pipeline([('scaling', preprocessing.StandardScaler()),
                     ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))])


def AdA_model():
    """
    Initial results were good but training and testing is a lot slower than the other models.
    """
    return AdaBoostClassifier()


# My own tester
def test_classifier(features, labels, clf, n_splits=2500, show_results=False):
    """
    The difference to the tester.py is that the metrics are calculated for every run an averaged at the end.
    """
    accu, r_precision, r_recall, r_f1 = [], [], [], []
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
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

    accuray = np.mean(accu)
    precision = np.mean(r_precision)
    recall = np.mean(r_recall)
    f1 = np.mean(r_f1)

    if show_results:
        print "Classifier:", clf
        print "Accuracy:", accuray, "Precision:", precision, \
            "Recall", recall, "F1", f1
        print ""

    return (accuray, precision, recall, f1)


# wrapper for the tester.py
def baseline_tester(features_list, classifier):
    print "features used:", features_list
    dump_classifier_and_data(classifier, my_dataset, features_list)
    tester.main()


# wrapper for my own tester
def baseline_own_testing(feat, cls, n_splits=1000, show_results=False):
    labels, features = targetFeatureSplit(featureFormat(my_dataset, feat,
                                                        sort_keys=True))
    return test_classifier(features, labels, cls, n_splits=n_splits, show_results=show_results)


def run_both_testers(feature_list, model):
    print "# Running a Model with both testers #"
    baseline_tester(feature_list, model)
    baseline_own_testing(feature_list, model, 1000, show_results=True)


# Function to add (and substract) features for search for best feature combination
def vary_features(input_features, classifier, testing_function=baseline_own_testing, show_results=False,
                  score="f1"):
    combinations_tested = []

    d = {"accuracy": 0,
         "precision": 1,
         "recall": 2,
         "f1": 3}

    target = d[score]

    if show_results:
        print "# Varying features #"
        print "features used:", len(input_features), " -> ", input_features
    combinations_tested.append((testing_function(input_features, classifier)[target], input_features))

    if show_results:
        print "## Elimiating ##"

    original = copy.copy(input_features)
    for i, feat in enumerate(original[1:]):
        features = copy.copy(original)
        if show_results:
            print "trying", feat
        features.pop(i + 1)
        if show_results:
            print "features used:", len(features), " -> ", features
        f1 = testing_function(features, classifier)[target]
        if f1 > 0:
            combinations_tested.append((f1, features))

    if show_results:
        print "## Adding ##"
    testing_function(original, classifier)
    for feat in all_features:
        if show_results:
            print "trying: ", feat
        features = copy.copy(original)
        if feat in features:
            continue
        else:
            features.append(feat)
        if show_results:
            print "features used:", len(features), " -> ", features
        f1 = testing_function(features, classifier)[target]
        if f1 > 0:
            combinations_tested.append((f1, features))

    return combinations_tested


# Recursive Wrapper for the varying function
def vary_recursive(feature_list, model, score="f1"):
    results = vary_features(feature_list, model, show_results=False, score=score)
    results_sorted = sorted(results, reverse=True)

    if results_sorted[0][1] == feature_list:
        print "best found, " + score, results_sorted[0]
        print "performace own tester"
        baseline_own_testing(results_sorted[0][1], model, show_results=True)
        print "performace tester.py"
        baseline_tester(results_sorted[0][1], model)

    else:
        print "trying another, " + score, results_sorted[0]
        vary_recursive(results_sorted[0][1], model)


# some tuning functions to find the best params for the models
def tuning_knn_scaled():
    """
    Best: k = 5, weights = 'uniform'
    """
    for n_ in range(1, 11):
        print n_, 'uniform'
        baseline_tester(best_knn_feature_list_scaled, KNN_model_scaled(n_, 'uniform'))
        print n_, 'distance'
        baseline_tester(best_knn_feature_list_scaled, KNN_model_scaled(n_, 'distance'))


def tuning_knn(feature_list):
    param_grid = {'knn__n_neighbors': [3, 4, 5, 6, 7], 'knn__weights': ['uniform', 'distance']}
    clf = GridSearchCV(KNN_model_scaled(), param_grid, scoring="f1")
    labels, features = targetFeatureSplit(featureFormat(my_dataset, feature_list, sort_keys=True))
    # print KNN_model_scaled().get_params().keys()
    clf.fit(features, labels)
    result = pd.DataFrame(clf.cv_results_)
    return result


def tuning_knn_not_scaled():
    """
    Best: k = 5, weights = 'distance'
    """
    for n2 in range(1, 11):
        print n2, 'uniform'
        baseline_tester(best_knn_feature_list_not_scaled, KNN_model_not_scaled(n2, 'uniform'))
        print n2, 'distance'
        baseline_tester(best_knn_feature_list_not_scaled, KNN_model_not_scaled(n2, 'distance'))


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
        for n_ in range(1, 10):
            print n_
            baseline_tester(best_dt_feature_list, DT_model(max_depth=n_))
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
    print "*** Final Scores - tester.py ***"
    print ""
    print "AdaBoost"
    baseline_tester(best_adaboost_feature_list, AdA_model())
    print ""
    print "Naive Bayes"
    baseline_tester(best_nb_feature_list, NB_model())
    print ""
    print "Desicion Tree"
    baseline_tester(best_dt_feature_list, DT_model())
    print ""
    print "K-Neigbohrs scaled"
    baseline_tester(best_knn_feature_list_scaled, KNN_model_scaled())
    print ""
    print "K-Neighbors not scaled"
    baseline_tester(best_knn_feature_list_not_scaled, KNN_model_not_scaled())


if False:
    run_both_testers(best_knn_feature_list_scaled, KNN_model_scaled(n_neighbors=3, weights='uniform'))
    # print "final + count nans"
    # run_both_testers(best_knn_feature_list_scaled + ['count_nans'], KNN_model_scaled(n_neighbors=3,
    # weights='uniform'))

if False:
    print "## Recursive Loop ##"
    vary_recursive(best_knn_feature_list_scaled, KNN_model_scaled(n_neighbors=3), score="f1")

if False:
    # tuning_knn_scaled()
    print tuning_knn(best_knn_feature_list_scaled)

if False:
    show_final_scores()

if True:
    print "# Submitting final model #"
    print KNN_model_scaled()
    dump_classifier_and_data(KNN_model_scaled(),
                             my_dataset,
                             best_knn_feature_list_scaled)

print "# End #"
