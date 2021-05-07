import pandas as pd 
from xgboost import XGBRegressor
import sklearn

class process(object):
    def __init__(self):
        pass
        
    def preprocess(self):
        self.train['am+pm'] = self.train['am'] + self.train['pm']
        self.train['year'] = self.train['datetime'].dt.year
        self.train['month'] = self.train['datetime'].dt.month
        self.train['day'] = self.train['datetime'].dt.day
        return self.train
class XGboostRegression(object):

    def XGboostRegression(self, param_grid, X_train, y_train):
        xgb = XGBRegressor()
        model_xgb = sklearn.model_selection.GridSearchCV(estimator = xgb, param_grid = param_grid, n_jobs = -1)
        model_xgb.fit(X_train, y_train)
        y_pred = model_xgb.predict(X_test)
        return y_pred