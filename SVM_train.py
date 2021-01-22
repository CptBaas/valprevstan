####
#Fall vs No-Fall SVM training
####

from sklearn import svm, preprocessing
import pandas
import pickle
import numpy as np
import imblearn as IB
from sklearn.metrics import confusion_matrix,recall_score, make_scorer
from sklearn.model_selection import GridSearchCV


def series_to_array(series):
    """
    iteratively convert df column to 1D-array

    Keyword argument:
    series: the series to convert (numpy.series)
    """
    temp_list = list()
    for i in series:
        if i == "Fall":
            temp_list.append(1)
        elif i == "O":
            temp_list.append(0)
        else:
            temp_list.append(i)
        
    return(np.asarray(temp_list))

#hardcoded resampling strategy
oversample = IB.over_sampling.SMOTE(sampling_strategy = {1:4531, 0: 179788},random_state = 314)
undersample = IB.under_sampling.RandomUnderSampler(sampling_strategy = {1: 4531, 0: 2266},random_state = 314)

steps = [("o", oversample), ("u", undersample)]
pipeline = IB.pipeline.Pipeline(steps = steps)
 
#load train and validation sets
train =  pickle.load(open("TrainStemAllNewModelAN.p","rb"))
val =  pickle.load(open("ValStemAllNewModelAN.p","rb"))

#split sets into X and y
X_train = series_to_array(train["Embedding"])
Y_train = series_to_array(train["Fall"])
X_val = series_to_array(val["Embedding"])
Y_val = series_to_array(val["Fall"])

X, y = pipeline.fit_resample(X_train,Y_train) 

#uncomment for gridsearch
"""
#param_grid = {"C":[1, 5, 10, 50], "gamma": [1, 0.5, 0.1, 0.05], "kernel": ["rbf"]}
param_grid = {"C":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], "gamma": [1.13,1.15,1.18,1.21,1.23,1.25,1.28]}

grid = GridSearchCV(svm.SVC(decision_function_shape = "ovr"), param_grid, refit = True,verbose =2, scoring = "precision", n_jobs = -1)
grid.fit(X,y)


pickle.dump(grid, open("estimator_grid_all.p", "wb"))
print(grid.best_estimator_)

print(grid.best_params_)

print(grid.best_score_)

"""

#define and train SVM. hardcoded parameters
clf = svm.SVC(decision_function_shape = "ovr", C = 0.6, kernel = "rbf", gamma = 1.15)
clf.fit(X, y)

#predict
result = clf.predict(X_val)

#save result and SVM
val["Prediction"] = result

pickle.dump(val, open("val_prediction_recall.p","wb"))
pickle.dump(clf, open("SVM_fall_recall.p","wb"))

#print result
matrix = confusion_matrix(Y_val, result, labels = [1,0])
print(matrix) 

