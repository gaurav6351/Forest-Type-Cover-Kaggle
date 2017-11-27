# Import the needed referances
import pandas as pd
import numpy as np
import csv as csv

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from itertools import combinations


#Shuffle the datasets
from sklearn.utils import shuffle
from numpy import array,array_equal

#Learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

train_dataset = pd.read_csv('/home/gaurav/forest-type/train.csv')
test_dataset = pd.read_csv('/home/gaurav/forest-type/test.csv')

train_dataset.dtypes.value_counts()

nas = pd.concat([train_dataset.isnull().sum(), test_dataset.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 

full_dataset = [train_dataset, test_dataset]
# Remove constant features

def identify_constant_features(dataframe):
    count_uniques = dataframe.apply(lambda x: len(x.unique()))
    constants = count_uniques[count_uniques == 1].index.tolist()
    return constants

constant_features_train = set(identify_constant_features(train_dataset))

train_dataset.drop(constant_features_train, inplace=True, axis=1)
test_dataset.drop(constant_features_train, inplace=True, axis=1)


from sklearn import preprocessing

colToScale=['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology', 
             'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
             'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
             'Horizontal_Distance_To_Fire_Points']

X_train = train_dataset.drop(["Cover_Type","Id"],axis=1)
Y_train = train_dataset["Cover_Type"]
X_test  = test_dataset.drop("Id",axis=1).copy()


#from sklearn.preprocessing import StandardScaler

#for col in colToScale:
 #   scaler = min_max_scaler.fit(ll_data[col].values.reshape(-1,1).astype('float_'))
  #  xtrain_scale[col] = scaler.transform(X_train[col].values.reshape(-1,1).astype('float_'))
   # test_scale[col] = scaler.transform(X_test[col].values.reshape(-1,1).astype('float_'))

from sklearn import cross_validation as cv
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV
forest = RandomForestClassifier()

parameter_grid = {'max_depth':[1,2,3,4,5],'n_estimators': [50,100,150,200,250],'criterion': ['gini','entropy']}

cross_validation = StratifiedKFold(Y_train, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(X_train, Y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


clf_rf = grid_search


ypred_rf=clf_rf.predict(X_test)

ids=test_dataset['Id']

pd.DataFrame({'Id':ids,'Cover_Type':ypred_rf},
            columns=['Id','Cover_Type']).to_csv('/home/gaurav/forest-type/o2.csv',index=False)

print(pd.Series(ypred_rf).value_counts(sort=False))


