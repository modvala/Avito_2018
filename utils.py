import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import TransformerMixin
from collections import defaultdict

class TargetEncoding(object):

    def __init__(self):
        self.voc = defaultdict()
    def fit(self, X, y, alpha = 0):
        
        target = y.name
        feature = X.name
        X = pd.concat([X, y], axis =1)
        # https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

        self.voc = X.groupby(feature)[target].agg(["mean", "count"])
        global_mean = X[target].mean()
        self.voc['__prior__'] = global_mean
        self.voc["mean"] = (self.voc["mean"].values*self.voc["count"].values+global_mean*alpha)/(self.voc["count"].values+alpha)
        self.voc['__prior__'] = (self.voc['__prior__']*self.voc["count"].sum()+global_mean*alpha)/(self.voc["count"].sum()+alpha) 

        return self

    def transform(self, X):
        
        return  X.map(self.voc['mean'].to_dict()).fillna(self.voc['__prior__'].unique()[0]).values



class gridsearch(object):

    def __init__(self, cv1=3, cv2=3):
        self.cv1 = cv1
        self.cv2 = cv2

    def gridsearch(self, X, y, alpha = 0):

            result = pd.DataFrame(np.zeros((X.shape[0], 9)))
            k = 0
            kf1 = KFold(n_splits=3, random_state=None, shuffle=False)
            kf2 = KFold(n_splits=3, random_state=None, shuffle=False)
            for train_index, test_index in kf1.split(X):
                for train_train, train_test in kf2.split(train_index):
                    train_train_index, train_test_index = train_index[train_train], train_index[train_test]
                    te = TargetEncoding()
                    te.fit(X[train_train_index], y[train_train_index], alpha = alpha)
                    result.loc[train_test_index, k] = te.transform(X[train_test_index])
                    result.loc[test_index, k] = te.transform(X[test_index])
                    k+=1
            
            return result




