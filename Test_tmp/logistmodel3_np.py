import numpy as np
def sigmoid(x):
    z = 1 / (1+np.exp(-x))
    return z
def initialize_params(dims):
    w = np.zeros((dims,1))
    b = 0
    return w,b
def logistic(X,y,W,b):
    num_train = X.shape[0]
    num_feature= X.shape[1]
    a = sigmoid(np.dot(X,W) + b)
    cost = -1/num_train*np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    dW= np.dot(X.T,(a-y))/num_train
    db = np.sum(a-y)/num_train
    cost = np.squeeze(cost)
    return a,cost,dW,db
def logistic_train(X,y,learning_rate,epochs):
    W,b=initialize_params(X.shape[1])
    cost_list=[]
    for i in range(epochs):
        a,cost,dW,db =logistic(X,y,W,b)
        W=W-learning_rate*dW
        b=b-learning_rate*db
        if i%100==0:
            cost_list.append(cost)
            print('epoch %d cost %f' % (i,cost))
    params = {
        'W': W,
        'b': b
    }
    grads = {
        'dw': dW,
        'db': db
    }
    return cost_list, params, grads
def predict(X,params):
    y_pred=sigmoid(np.dot(X,params['W'])+params['b'])
    for i in range(len(y_pred)):
        if y_pred[i]>0.5:
            y_pred[i]=1
        else:
            y_pred[i]=0
    return y_pred
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_classification
X,labels = make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2,
                               random_state=1,n_clusters_per_class=2)
rng = np.random.RandomState(2)
X+=2*rng.uniform(size=X.shape)
unique_labels=set(labels)
colors=plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
for k,col in zip(unique_labels,colors):
    x_k=X[labels==k]
    plt.plot(x_k[:,0],x_k[:,1],'o',
             markerfacecolor=col,
             markeredgecolor='k',
             markersize=14)
    plt.title('Simulated binary data set')
    plt.show()
offset =int(X.shape[0] * 0.9)
X_train,y_train=X[:offset],labels[:offset]
X_test,y_test=X[offset:],labels[offset:]
y_train =y_train.reshape((-1,1))
y_test =y_test.reshape((-1,1))

cost_list,params,grads = logistic_train(X_train,y_train,0.01,1000)
print(params)
y_pred=predict(X_test,params)
print(y_pred.T)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

def plot_decision_boundary(X_train,y_train,params):
    n=X_train.shape[0]
    xcord1=[]
    ycord1=[]
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if y_train[i]==1:
            xcord1.append(X_train[i][0])
            ycord1.append(X_train[i][1])
        else:
            xcord2.append(X_train[i][0])
            ycord2.append(X_train[i][1])
    fig =plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=32,c='red')
    ax.scatter(xcord2, ycord2, s=32, c='green')
    x=np.arange(-1.5,3,0.1)
    y=(-params['b']-params['W'][0]*x)/params['W'][1]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.xlabel('X2')
    plt.show()
plot_decision_boundary(X_train,y_train,params)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0).fit(X_train,y_train.ravel())
y_pred=clf.predict(X_test)
print(y_pred)