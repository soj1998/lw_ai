from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle

diabetes = load_diabetes()
data, target = diabetes.data,diabetes.target
X,y=shuffle(data,target,random_state=13)
offset =int(X.shape[0] * 0.8)
X_train,y_train=X[:offset],y[:offset]
X_test,y_test=X[offset:],y[offset:]
#y_train =y_train.reshape((-1,1))
#y_test =y_test.reshape((-1,1))
regr= linear_model.LogisticRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)
print("Mean squared error:%.2f" % mean_squared_error(y_test,y_pred))
print("R score:%.2f" % r2_score(y_test,y_pred))