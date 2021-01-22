####
#In vs Out SVM training
####
from sklearn import svm, preprocessing
import pandas
import pickle
import numpy as np
import imblearn as IB
from sklearn.metrics import confusion_matrix,recall_score, precision_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from collections import Counter

#hardcoded resampling strategy
oversample = IB.over_sampling.SMOTE(sampling_strategy = {"in":1451, "out": 3423},random_state = 314)
undersample = IB.under_sampling.RandomUnderSampler(sampling_strategy = {"in": 1451, "out": 2902},random_state = 314)

steps = [("o", oversample), ("u", undersample)]
pipeline = IB.pipeline.Pipeline(steps = steps)


def series_to_array(series):
    """
    iteratively convert df column to 1D-array

    Keyword argument:
    series: the series to convert (numpy.series)
    """
    temp_list = list()
    for i in series:
        temp_list.append(i)
        
    return(np.asarray(temp_list))


def filter_df(df):
    """
    returns a dataframe consisting of only rows that have a fall mentioned

    Keyword argument:
    df: the dataframe to filter (DataFrame)
    """
    df = df.loc[df["InOut"].isin(["in","out","unpec"])]
    df = df.replace(to_replace = "unspec", value = "in")
    
    
    return(df)
 
#load in the training and validation sets
train =  pickle.load(open("TrainStemAllNewModelAN.p","rb"))
val =  pickle.load(open("TestStemAllNewModelAN.p","rb"))

#filter the sets
train = filter_df(train)
val = filter_df(val)

#split sets into X and y
X_train = series_to_array(train["Embedding"])
Y_train = series_to_array(train["InOut"])
X_val = series_to_array(val["Embedding"])
Y_val = series_to_array(val["InOut"])

X, y = pipeline.fit_resample(X_train,Y_train) 

#uncomment for gridsearch
"""
param_grid = {"C":[1,10,50,100,150], "gamma": [0.1,0.3,0.5,0.8,1.1,1.3,1.5,2,5,10]}

recall_scorer = make_scorer(recall_score, pos_label = "in")

grid = GridSearchCV(svm.SVC(decision_function_shape = "ovr"), param_grid, refit = True,verbose =2, scoring = recall_scorer, n_jobs = -1)
grid.fit(X,y)

pickle.dump(grid, open("estimator_grid_balanced_lowered_stopword_invsout.p", "wb"))

print(grid.best_estimator_)

print(grid.best_params_)

print(grid.best_score_)

"""

#define and train SVM. hardcoded parameters
clf = svm.SVC(decision_function_shape = "ovr", C = 106, kernel = "rbf", gamma = 1.27)
clf.fit(X, y)

#predict
result = clf.predict(X_val)

#save result and SVM
val["Prediction"] = result

#pickle.dump(val, open("val_prediction_all_SVM2.p", "wb"))
#pickle.dump(clf, open("SVM_inout_precision.p", "wb"))
 
matrix = confusion_matrix(Y_val, result, labels = ["in","out"])
print(matrix) 

print(precision_recall_fscore_support(Y_val,result,labels = ["in", "out"]))



 
 
