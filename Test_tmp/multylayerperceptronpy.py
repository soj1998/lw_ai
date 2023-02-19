import numpy as np
import matplotlib.pyplot as plt


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h = 4
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2
                  }
    return parameters


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def forward_propagation(X, parameters):
    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    z1 = np.dot(w1,X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    cache = {'z1': z1,
             'a1': a1,
             'z2': z2,
             'a2': a2}
    return a2, cache


def compute_cost(a2, Y):
    m = Y.shape[1]
    logprobs = np.log(a2)*Y + np.log(1-a2)*(1-Y)
    cost = -1/m * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    w1 = parameters['w1']
    w2 = parameters['w2']
    a1 = cache['a1']
    a2 = cache['a2']
    dz2 = a2 - Y
    dw2 = 1/m * np.dot(dz2, a1.T)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)  # axis=1 第二个维度求和 keepdims 保持矩阵维度特性
    dz1 = np.dot(w2.T, dz2)*(1-a1**2)
    dw1 = 1/m * np.dot(dz1, X.T)
    db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)
    grads = {'dw1': dw1,
             'dw2': dw2,
             'db1': db1,
             'db2': db2}
    return grads


def update_parameter(parameters, grads, learning_rate=1.2):
    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    dw1 = grads['dw1']
    dw2 = grads['dw2']
    db1 = grads['db1']
    db2 = grads['db2']
    w1 -= dw1 * learning_rate
    w2 -= dw2 * learning_rate
    b1 -= db1 * learning_rate
    b2 -= db2 * learning_rate
    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2
                  }
    return parameters


def nn_model(X, Y, n_h, num_iteration = 10000,print_cost=False):
    np.random.seed(3)  # 参数比喻成“堆”；eg. seed(5)：表示第5堆种子
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    for i in range(0, num_iteration):
        a2, cache = forward_propagation(X, parameters)
        cost = compute_cost(a2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameter(parameters, grads, learning_rate=1.2)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i : %f" % (i, cost))
        return parameters


def create_dataset():
    np.random.seed(1)
    m = 400
    N = int(m/2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4
    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12, N) + np.random.randn(N)*0.2
        r = a*np.sin(4*t) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2>0.5)
    return predictions


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


X, Y = create_dataset()
parameters1 = nn_model(X, Y, n_h =4, num_iteration=10000, print_cost=True)
predictions1 = predict(parameters1, X)
print('Accuracy: %d' % float((np.dot(Y, predictions1.T) +
      np.dot(1-Y, 1-predictions1.T))/float(Y.size)*100) + '%')


plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
plt.show()

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h =4, num_iteration=10000, print_cost=True)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0])
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
