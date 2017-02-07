1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of 
your answer, give some background on the dataset and how it can be used to answer the project question. Were there any 
outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, 
“outlier investigation”]

* The goal of the project is to identify possible person of interest ('poi') from a dataset of persons involved in the 
fraud related bankruptcy of Enron. As the exact definition of a 'poi' is not available, machine learning might be able 
to find structures that point towards poi. 
* I found one outlier in the dataset, that was a summary line ('TOTAL') and therefore removed. Another outlier was a 
line that pointed towards a business. I left those untouched since they might very well be of importance.
* **Overview**:
* The dataset contained 145 data points, 18 were labeled as POI, 127 as Not-POI. The original dataset contained 20 
different features, but I did not use the email feature. Some additional features were created from these features. 
Features had a NaN if no data was available. Some features had very little information available. The three features
   with the most NaNs were 'restricted_stock_deferred' (88% NaNs), 'loan_advances' (97% NaNs) and 'director_fees' 
    (89% NaNs).

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? 
Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own 
feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale 
behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature 
selection step, if you used an algorithm like a decision tree, please also give the feature importance of the features 
that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores 
and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly 
scale features”, “intelligently select feature”]

* Models were tested with a set of 6 different features that were found by using correlations. The limit was set at an 
absolute correlation of 0.26. This was set to find a reasonable big list of features to start with that is not too big. 
NaNs were treated as 0 to compute correlation. Correlation was used as a basic measure to find features that might 
hold predictive value towards the label.

* I started with these six features (correlation):

    exercised_stock_options    0.388241

    total_stock_value          0.384127

    bonus                      0.360262

    salary                     0.341365

    count_nans                -0.269021

    deferred_income           -0.275364

* After starting with these features I tried to find a better combination of features. This was done with a recursive 
function removing and adding features and testing the performance. Within every execution of the function each feature
of the starting list was removed and the performance tried for the remaining features. Then every feature not included
in the list was then added (only one feature at the time) and the performance measured. I not combination returned a 
better F1 score the best feature list was returned. If a better was found the function started again with the new 
feature list.

* For the scaled KNN Model the following combinations were used to seed the function. All other combinations that 
either remove or add one feature were tried. After 8 cycles I could improve the F1 score from 0.266 to 0.432.

trying another - (0.26671777296777299, ['poi', 'bonus', 'count_nans', 'deferred_income', 'exercised_stock_options', 
'total_stock_value'])

trying another - (0.30042224535895423, ['poi', 'bonus', 'count_nans', 'deferred_income', 'exercised_stock_options', 
'total_stock_value', 'deferral_payments'])

trying another - (0.32304669750712922, ['poi', 'bonus', 'deferred_income', 'exercised_stock_options', 
'total_stock_value', 'deferral_payments'])

trying another - (0.42032529797766816, ['poi', 'bonus', 'exercised_stock_options', 'total_stock_value', 
'deferral_payments'])

trying another - (0.4269693717931603, ['poi', 'bonus', 'exercised_stock_options', 'total_stock_value', 
'deferral_payments', 'restricted_stock_deferred_discrete'])

trying another - (0.43067470373527755, ['poi', 'bonus', 'exercised_stock_options', 'total_stock_value', 
'deferral_payments', 'restricted_stock_deferred_discrete', 'from_this_person_to_poi'])

trying another - (0.43198028675735678, ['poi', 'bonus', 'exercised_stock_options', 'total_stock_value', 
'deferral_payments', 'restricted_stock_deferred_discrete', 'from_this_person_to_poi', 'restricted_stock_deferred'])

best found - (0.43198028675735678, ['poi', 'bonus', 'exercised_stock_options', 'total_stock_value', 'deferral_payments', 
'restricted_stock_deferred_discrete', 'from_this_person_to_poi', 'restricted_stock_deferred'])

* After this feature selection the model was tuned for different parameters. For the K-Neighbors the number of Neighbors 
and the measurement of distance was searched. Since n_neigbors = 3 showed better results the feature selection was 
tried again starting at the last list but could not improve the 

* Final Score: best found, f1 (0.53099817270696947, ['poi', 'bonus', 'exercised_stock_options', 'total_stock_value', 
'deferral_payments', 'restricted_stock_deferred_discrete', 'restricted_stock_deferred'])

* Further tuning did not improve the performance of the model.

* I added several new features. The most important was 'count_nans' with just counted the 'NaN's in a row. The idea is, 
that more or less data about a person might be correlated to the status as poi. The feature hold a correlation of -27% 
and therefore is in the top six of absolute correlations and was used to start the recursive feature selection. The 
feature got removed from the feature list during the recursive feature selection. If added to the final feature list
the performance on the precision/recall/f1 drops considerably (tester.py): Final List 0.77 / 0.50 / 0.61; Final List + 
'count_nans' 0.67 / 0.32 / 0.44.
* Some discrete features were added where a feature had a lot of NaNs. Simplified information might hold more relevance.

3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between 
algorithms?  [relevant rubric item: “pick an algorithm”]

* My final model is K-Neighbors with a standard scaling applied. 

* I tried Naive Bayes, Decision Tree, AdABoost and K-Neighbors models. Naive Bayes, DT and AdaBoost (with DT as 
underlying model) do not use any measure of distance for the prediction. Therefore, the scale of the features doesn't 
need to be adjusted.

* K-Neighbors measures the distance between points. Therefor large features can dominate small features. Because of this 
scaling of the features usually should by applied. 

* Although, it might be the case that the bigger features are more important and therefore improve the model if not 
scaled. The unscaled algorithm was able to reach comparable scores.

* AdABoost performed considerable slower than the other models.

* All models could reach the required specs (Precision 0.3 +, Recall 0.3 +, F1 0.3 +):

 | Score     | AdABoost | Naive Bayes | Decision Tree | K-Neighbors | K-Neighbors (not scaled) |
|-----------|----------|-------------|---------------|-------------|--------------------------|
| Accuracy  | 0.87893  | 0.87273     | 0.85407       | 0.90736     | 0.86969                  |
| Precision | 0.55808  | 0.53031     | 0.48941       | 0.77101     | 0.63235                  |
| Recall    | 0.44200  | 0.39800     | 0.4970        | 0.50000     | 0.36550                  |
| F1        | 0.49330  | 0.45473     | 0.49318       | 0.60661     | 0.46324                  |


 
*** **Details: Final Scores - tester.py** ***

**AdaBoost**

features used ['poi', 'deferred_income', 'count_nans', 'salary', 'exercised_stock_options', 'expenses', 'to_messages', 
'shared_receipt_with_poi', 'other', 'from_poi_to_this_person']

AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
          
Accuracy: 0.87893  Precision: 0.55808 Recall: 0.44200    F1: 0.49330    F2: 0.46119

   Total predictions: 15000   True positives:  884   False positives:  700  False negatives: 1116  True negatives: 12300


**Naive Bayes**

features used ['poi', 'deferred_income', 'total_stock_value', 'salary', 'exercised_stock_options', 'expenses']

GaussianNB(priors=None)

Accuracy: 0.87273  Precision: 0.53031 Recall: 0.39800    F1: 0.45473    F2: 0.41890

   Total predictions: 15000   True positives:  796   False positives:  705  False negatives: 1204  True negatives: 12295


**Decision Tree**

features used ['poi', 'deferred_income', 'exercised_stock_options', 'expenses', 'deferral_payments', 'director_fees']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
            
Accuracy: 0.85407  Precision: 0.48941 Recall:0.4970   F1: 0.49318    F2: 0.49546

   Total predictions: 14000   True positives:  994   False positives: 1037  False negatives: 1006  True negatives: 10963


**K-Neighbors scaled**

features used ['poi', 'bonus', 'exercised_stock_options', 'total_stock_value', 'deferral_payments', 
'restricted_stock_deferred_discrete', 'restricted_stock_deferred']

Pipeline(steps=[('scaling', StandardScaler(copy=True, with_mean=True, with_std=True)), ('knn', 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='uniform'))])

Accuracy: 0.90736  Precision: 0.77101 Recall: 0.50000    F1: 0.60661    F2: 0.53781

   Total predictions: 14000   True positives: 1000   False positives:  297  False negatives: 1000  True negatives: 11703


**K-Neighbors not scaled**

features used ['poi', 'bonus', 'exercised_stock_options', 'salary', 'loan_advances_discrete']

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='distance')

Accuracy: 0.86969  Precision: 0.63235 Recall: 0.36550    F1: 0.46324    F2: 0.39919

   Total predictions: 13000   True positives:  731   False positives:  425  False negatives: 1269  True negatives: 10575

4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  H
ow did you tune the parameters of your algorithm? (Some algorithms do not have parameters that you need to tune -- if 
this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was 
not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  
[relevant rubric item: “tune the algorithm”]

* Tuning is the process of adjusting the parameters of an algorithm to improve the predictions. While tuning the model, 
mathematical model is varied to find a variation that fits better. 

* K-Neighbors was tuned on the count of neighbors and the weighting of the neighbors. This modifies the number of 
neighbors used and how the distance is accounted for. Both K-Neighbors algorithms performed best on n = 3 neighbors.
For the scaled features 'uniform' weighting did best, for the unscaled 'distance'.

5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  
[relevant rubric item: “validation strategy”]

* Validation is used to test if the model is doing well on data not know to the model, means data it was not trained on.
* Validation was difficult because of the small set of positive poi's (just 18 in the dataset). I therefore run the 
model often (1,000 times) and averaged the scores. Besides this I used the slightly different tester.py script to 
validate my models.

6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your 
metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of 
evaluation metrics”]

* Accuracy: The rate of true (positive and negative) predictions compared to all predictions.
* Precision: The rate of true positives predicted compared to all positively predicted elements.
* Recall: The rate of true positives predicted to all positive data points.
* F1-Score: The harmonic mean of precision and recall.
* My model was able to reach the following metrics (tester.py): 
Accuracy: 0.90736  Precision: 0.77101 Recall: 0.50000   F1: 0.60661    F2: 0.53781
* The high accuracy score of 90 % is only a bit better than pure chance, since about 85 % of the dataset is negative. 
The recall of 50 % means, that about 50 % of the relevant poi's will be found by the model. Of those found about 77 % 
are correct predictions ('precision'). 