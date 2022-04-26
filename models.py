
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge


from utils import Utils

class Models:

    def __init__(self):

        self.reg = {
            'SVR':SVR(),
            'GRADIENT' : GradientBoostingRegressor(),
            'Ridge' : Ridge()
        }

        self.params = {
            'SVR': {
                'kernel': ['linear','poly','rbf'],
                'gamma':['auto','scale'],
                'C' :[1,5,10]

            },'GRADIENT' : {
                'loss' : ['ls','lad'],
                'learning_rate' : [0.01,0.05,0.1]

            }, 'Ridge': {
                'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        }
    
    def gid_training(self,X,y):

        best_score = 99
        best_model = None

        for name, reg in self.reg.items():
            

            grid_reg = GridSearchCV(reg,self.params[name],scoring ='neg_mean_squared_error', cv = 5)
            grid_reg.fit(X,y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:

                best_score = score
                best_model = grid_reg.best_estimator_
                Scor = grid_reg.scorer_
        
        utils =Utils()
        utils.model_export(best_model,best_score,Scor)
