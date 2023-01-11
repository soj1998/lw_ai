import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
def linear_loss(X, y, w, b):
    num_train = X.shape[0]
    num_featurn = X.shape[1]
    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat -y)**2) / num_train
    dw = np.dot(X.T,(y_hat-y))/ num_train
    db = np.sum((y_hat-y))/ num_train
    return y_hat,loss,dw,db
def initialize_params(dims):
    w=np.ones((dims,1))
    b=0
    return w,b
def linear_train(X,y,learning_rate=0.01,epochs=10000):
    loss_his=[]
    w,b=initialize_params(X.shape[1])
    for i in range(1,epochs):
        y_hat,loss,dw,db=linear_loss(X,y,w,b)
        w+=-learning_rate*dw
        b += -learning_rate*db
        loss_his.append(loss)
        if i%10000 ==0:
            print('epoch %d loss %f' % (i,loss))
        params={
            'w':w,
            'b':b
        }
        grads = {
            'dw':dw,
            'db':db
        }
    return loss_his,params,grads

diabetes = load_diabetes()
data, target = diabetes.data,diabetes.target
X,y=shuffle(data,target,random_state=13)
offset =int(X.shape[0] * 0.8)
X_train,y_train=X[:offset],y[:offset]
X_test,y_test=X[offset:],y[offset:]
y_train =y_train.reshape((-1,1))
y_test =y_test.reshape((-1,1))
print("X_train's shape",X_train.shape)
print("X_test's shape",X_test.shape)
print("y_train's shape",y_train.shape)
print("y_test's shape",y_test.shape)
loss_his,params,grads = linear_train(X_train,y_train,1,200000)
print(params)

def predict(X, params):
    w= params['w']
    b = params['b']
    y_pred= np.dot(X,w) + b
    return y_pred
y_pred = predict(X_test,params)
def r2_score(y_test,y_pred):
    y_avg=np.mean(y_test)
    ss_tot=np.sum((y_test-y_avg)**2)
    ss_res = np.sum((y_test-y_pred)**2)
    r2 = 1-(ss_res/ss_tot)
    return r2

print(r2_score(y_test,y_pred))