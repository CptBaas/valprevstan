####
#In vs Out SVM prediction
####
from sklearn import svm, preprocessing
import pandas
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter

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
    df = df.loc[df["Prediction"] == 1].copy()
    return(df)

def replace_unspec(df):
    """
    returns a dataframe in which the unspecified locations are tranformed to in

    Keyword argument:
    df: the dataframe to convert (DataFrame)
    """
    df = df.replace(to_replace = "unspec", value = "in")
    return(df)

#load the validation set and the SVM
clf = pickle.load(open("SVM_inout_recall.p","rb"))
val = pickle.load(open("test_prediction_recall.p", "rb"))

#filter df and replace unspec
filtered_val = filter_df(val)
replaced_val = replace_unspec(val)

#split sets into X and y
X_val = series_to_array(filtered_val["Embedding"])
Y_val = series_to_array(filtered_val["InOut"])

#predict
result = clf.predict(X_val)

#save result
filtered_val["InOutPred"] = result

#Add predictions to the original dataframe
indices = list()

for row in filtered_val.index:
    while row != len(indices):
        indices.append("O")
    else:
        indices.append(filtered_val["InOutPred"][row])

while len(indices) < len(val):
    indices.append("O")
 
#save result again, this time in the whole dataset
replaced_val["InOutPred"] = indices



#print result
matrix = confusion_matrix(replaced_val["InOut"], indices, labels = ["in","out","O"])
print(matrix)

#save result
#pickle.dump(replaced_val, open("final_result_all.p","wb"))

 
