import numpy as np


def sign(x,w,b):
    return np.dot(x,w)+b


def initialize_parameters(dim):
    w=np.zeros(dim)
    b=0.0
    return w,b


def perceptron_train(X_train,y_train,learning_rate):
    w,b=initialize_parameters(X_train.shape[1])
    is_wrong=False
    while not is_wrong:
        wrong_count = 0
        for i in range(len(X_train)):
            x=X_train[i]
            y=y_train[i]
            if y*sign(x,w,b)<=0:
                w=w+learning_rate*np.dot(y,x)
                b=b+learning_rate*y
                wrong_count+=1
            if wrong_count==0:
                is_wrong=True
                print("没有错误点")
        params = {'w':w,'b':b}
    return params


import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
df =pd.DataFrame(iris.data,columns=iris.feature_names)
df['label']=iris.target
df.columns=['sepal length','sepal width','petallength','petalwidth','label']
data =np.array(df.iloc[:100,[0,1,-1]])
x,y = data[:,:-1],data[:,-1]
y=np.array([ 1 if i==1 else -1 for i in y])
params = perceptron_train(x,y,0.01)
print(params)



x_points = np.linspace(4, 7, 10)
y_hat = -(params['w'][0]*x_points + params['b'])/params['w'][1]
plt.plot(x_points, y_hat)

plt.plot(data[:50, 0], data[:50, 1], color='red', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], color='green', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()