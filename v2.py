
import pandas as pd
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd

# Load the training and test data sets
train = pd.read_csv('/home/gaurav/forest-type/train.csv')
test = pd.read_csv('/home/gaurav/forest-type/test.csv')

# Create numpy arrays for use with scikit-learn
train_X = train.drop(['Id','Cover_Type',],axis=1).values
train_y = train.Cover_Type.values
test_X = test.drop('Id',axis=1).values


X,X_,y,y_ = cross_validation.train_test_split(train_X,train_y,test_size=0.2)

pca = PCA()

pca.fit(train_X)
var= pca.explained_variance_ratio_

print var1

pca = PCA(n_components=8)
pca.fit(train_X)
X1=pca.fit_transform(train_X)

X=X1
rf = ensemble.RandomForestClassifier()
rf.fit(X,train_y)
y_rf = rf.predict(test)
#print(metrics.classification_report(y_,y_rf))
#print(metrics.accuracy_score(y_,y_rf))


# Write to CSV
pd.DataFrame({'Id':test.Id.values,'Cover_Type':y_test_rf})\
            .sort_index(ascending=False,axis=1).to_csv('/home/gaurav/forest-type/04.csv',index=False)
