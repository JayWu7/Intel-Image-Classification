import numpy as np
import matplotlib.pyplot as plt
from upload_data import upload_img_data, processing_image_matrix
import scipy.linalg.blas as blas
import sys, getopt


def sigmoid(z):
    '''
    implementing sigmoid function
    :param z: numpy array of scalar
    :return: sigmoid(z)
    '''
    A = 1 / (1 + np.exp(-z))
    cache = z
    return A, cache


def relu(z):
    '''
    Implement of the RELU function
    :param z: output of the linear layer
    :return:
            A -- post-activation parameter, of the same shape as Z
            cache -- a python dictionary containing 'Z'
    '''
    A = np.maximum(0, z)
    assert A.shape == z.shape

    cache = z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def layer_sizes(X, n_h):
    '''
    define the size of neural network layer
    :param X: input dataset of shape(input size, number of samples)
    n_h: a list, containing the size of each hidden layer
    :return:
    n_x -- the size of input layer
    n_h -- the size of hidden layer
    n_y -- the size of output layer
    '''
    n_x = X.shape[0]
    n_y = 1
    dims = [n_x] + n_h + [n_y]
    return dims


def initialize_parameters_deep(layer_dims, seed=16):
    '''
    initialize the parameters we used in this model
    :param layer_dims: python array containing the dimensions of each layer
    :return:
            parameters -- python dict containing our parameters
            seed -- the seed of random generator function
    '''
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]).astype(dtype='float32') * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)).astype(dtype='float32')
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    '''
    Implement the linear part of a layer's forward propagation.
    :param A: activations from previous layer
    :param W: weight matrix
    :param b: bias vector
    :return:
            Z -- the input of the activation function
            cache -- a python tuple containing 'A', 'W', 'b'
    '''
    # Z = np.dot(W, A) + b
    Z = blas.sgemm(alpha=1.0, a=W, b=A) + b

    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, w, b, activation):
    '''
    Implement the forward propagation for the Linear->Activation layer

    :param A_prev: activations from previous layer
    :param w: weights matrix
    :param b: bias vector
    :param activation: the activation method to be used in this layer, a string
    :return:
            A -- the output of the activation function
            cache -- a python tuple containing 'linear_cache' and 'activation_cache'
    '''

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, w, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, w, b)
        A, activation_cache = relu(Z)

    else:
        print('Activation is wrong, please check it as one of ["sigmoid", "relu"]')
        return None

    assert (A.shape == (w.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    '''
    Implement forward propagation for our L layers model.
    :param X: input data, numpy array of shape (input size, number of examples)
    :param parameters:  output of the initialize_parameters_deep()
    :return:
            AL -- last post-activation value
            caches -- list of caches containing:
                            every cache of linear_activation_forward()
    '''

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches


def compute_cost(AL, Y):
    '''
    Implement the cost function
    :param AL: probability vector corresponding to our label predictions
    :param Y: true 'label' vector
    :return:
            cost -- cross-entropy cost
    '''

    m = Y.shape[1]

    logprobs = np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))
    cost = (-1 / m) * np.sum(logprobs)
    cost = float(cost)
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    '''
    Implement the linear portion of backward propagation for a single layer.
    :param dZ:  Gradient of the cost with respect to the linear output
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation
    :return:
            dA_prev -- Gradient of the cost with respect to the activation
            dW -- Gradient of the cost with respect to W
            db -- Gradient of the cost with respect to b
    '''

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    # dW = (1 / m) * blas.sgemm(alpha=1.0, a=dZ, b=A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    # dA_prev = blas.sgemm(alpha=1.0, a=W.T, b=dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    '''
    Implement the backward propagation for the Linear -> Activation layer.
    :param dA: post-activation gradient for current layer 1
    :param cache: tuple of values (linear_cache, activation_cache)
    :param activation: the activation to be used in this layer
    :return:
            dA_prev -- Gradient of the cost with respect to the activation
            dW -- Gradient of the cost with respect to W
            db -- Gradient of the cost with respect to b
    '''

    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    else:
        print('Activation is wrong, please check it as one of ["sigmoid", "relu"]')
        return None

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    '''
    Implement the backward propagation for the Linear -> Relu * (L-1) -> Sigmoid group
    :param AL: probability vector, output of the forward propagation
    :param Y: true label vector
    :param caches: list of caches including:
                        every cache of linear_activation_forward() with relu and sigmoid
    :return:
            grads -- A python dict with the gradients of dA. dW, and db.
    '''

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      'sigmoid')
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate=0.5, num_iterations=1000, seed=16, print_cost=True):
    '''
    Implements a L-layer neural network

    :param X: input data, numpy array of shape (num_px * num_px * 3, number of examples)
    :param Y: true 'label' vector
    :param layer_dims: list containing the input size and each layer size.
    :param learning_rate: learning rate of the gradient descent
    :param num_iterations: number of iterations of the optimization loop
    :param seed: seed of random generator
    :param print_cost: if prints the cost every 100 steps
    :return:
            parameters -- parameters learnt by the model, which can nbe used in the prediction.
    '''

    costs = []
    parameters = initialize_parameters_deep(layer_dims)

    if num_iterations is None:
        num_iterations = 1000
    if seed is None:
        seed = 16
    if learning_rate is None:
        learning_rate = 0.5

    import time
    for i in range(0, num_iterations):
        t = time.time()
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1 == 0:
            print('Cost after iteration %i: %f' % (i, cost))
            costs.append(cost)
        print(time.time() - t)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundred)')
    plt.title('Learning rate = {}'.format(learning_rate))
    plt.show()

    return parameters


def predict(parameters, X):
    '''
    Using the learned parameters, predicts a class for each sample in X
    :param parameters: python dict containing our parameters
    :param X: input data
    :return:
        predictions -- vector of predictions of our model (0:sushi, 1:sandwich)
    '''
    AL, cache = L_model_forward(X, parameters)
    predictions = [1 if i > 0.5 else 0 for i in np.squeeze(AL)]
    predictions = np.array(predictions)

    return predictions


def main(argv):
    x, y = upload_img_data()
    X, Y = processing_image_matrix(x, y)
    layers_dims = layer_sizes(X, [20, 7, 5, 3])

    try:
        opts, _ = getopt.getopt(argv, 'n:s:a:', ['num_iterations=', 'seed=', 'alpha='])
    except getopt.GetoptError:
        print('get parameters error, using the default parameters!')

    num_iterations, seed, alpha = None, None, None
    for opt, arg in opts:
        if opt == '-n' or opt == '--num_iterations':
            num_iterations = int(arg)
        elif opt == '-s' or opt == '--seed':
            seed = int(arg)
        elif opt == '-a' or opt == '--alpha':  # learning rate
            alpha = float(arg)
        else:
            pass

    parameters = L_layer_model(X, Y, layers_dims, num_iterations=num_iterations, learning_rate=alpha, seed=seed)
    predictions = predict(parameters, X)

    print("predictions mean = " + str(np.mean(predictions)))
    print('Accuracy: %d' % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')


if __name__ == '__main__':
    main(sys.argv[1:])
