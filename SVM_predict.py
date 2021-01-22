####
#Fall vs No-Fall SVM prediction
####
from sklearn import svm, preprocessing
import pandas
import pickle
import numpy as np 
from sklearn.metrics import confusion_matrix
 
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


#load in the validation or test data. hardcoded
val =  pickle.load(open("TestStemAllNewModelAN.p","rb"))
#load in the SVM
clf = pickle.load(open("SVM_fall_recall.p","rb"))

#split validation set into X and y
X_val = series_to_array(val["Embedding"])
Y_val = series_to_array(val["Fall"])

#predict
result = clf.predict(X_val)

#save the result for later analyses
val["Prediction"] = result
pickle.dump(val, open("test_prediction_recall.p","wb"))

#print the result
matrix = confusion_matrix(Y_val, result, labels = [1,0])
print(matrix) 
