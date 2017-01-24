1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

* The goal of the project is to identify possible person of interest ('poi') from a dataset of persons involved in the fraud related bancrupcy of enron. As the exact defintion of a 'poi' is not available, machine learning might be able to find structures that point towards poi. 
* I found one outlier in the dataset, that was a summary line ('TOTAL') and therefore removed. Another outlier was a line that pointed towards a business. I left those untouched since they might very well be of importance.

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

* I used the features ['poi', 'bonus', 'salary', 'count_nans', 'exercised_stock_options', 'deferral_payments', 'other'] for my final classifier. I generated a list of some features from basic correlation to start with. I then removed and added a feature until I could not improve anymore. This was done by some looping functions. 
 * I tried scaling on the k-neighbors algorithm, but wasn't able to tune it better than without scaling.
 * I added several new features. The most important was 'count_nans' with just counted the 'NaN's in a row. The idea is, that more or less data about a person might be correlated to the status as poi.
 * Some discrete features were added where a feature had a lot of NaNs. Simplified imformation might held more relevance.

3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

* K-Neighbors with unscaled featured performed best. I tried Naive Bayes, Decision Tree and K-Neighbors with scaled data.
* All model were able to reach the required specs (Precision 0.3 + , Recall 0.3 +, F1 0.3 +), but the K-Neighbors 
outperformed the other by several percentage points.

4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

* K-Neighbors was tuned on the count of neighbors and the weighting of the neighbors. This modifies the number of 
neighbors used and how the distance is accounted for. Both K-Neighbors algorithms performend best on n = 5 neighbors.
For the scaled features 'uniform' weighting did best, for the unscaled 'distance'.

5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

* ...

6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

* Accuracy: 0.90427	Precision: 0.73818	Recall: 0.43700	F1: 0.54899	F2: 0.47583
	Total predictions: 15000	True positives:  874	False positives:  310	False negatives: 1126	True negatives: 12690