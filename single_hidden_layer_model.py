import numpy as np
import matplotlib.pyplot as plt
import sklearn
from upload_data import upload_img_data, processing_image_matrix
from PIL import Image
import shutil
import scipy.linalg.blas as blas
import sys, getopt


# def processing_img(directory='./dataset/train/sushi/'):
#     res = []
#     for im_name in listdir(directory):
#         if not im_name.startswith('.'):  # ignore the hidden file in the directory
#             filename = directory + im_name
#             im = Image.open(filename)
#             if im.size == (300, 300):
#                 shutil.copy(filename, './dataset/train/data')
#
#
# processing_img()


# upload the data from dataset

# def upload_img_data(directory='./dataset/train/data'):
#     '''
#     read the images and return the matrix of these images
#     :param directory: directory which store these images
#     :return:  numpy arrays with shape(m, 300,300,3)
#     '''
#     images_name = [n for n in listdir(directory) if not n.startswith('.')]
#     images = np.ndarray((len(images_name), 300, 300, 3))
#     for i, na in enumerate(images_name):
#         images[i] = plt.imread('{}/{}'.format(directory, na), 'JPG')
#     print(images.shape)
#     return images


def sigmoid(z):
    '''
    implementing sigmoid function
    :param z: numpy array of scalar
    :return: sigmoid(z)
    '''
    s = 1 / (1 + np.exp(-z))
    return s


# define the neural network structure
def layer_sizes(X):
    '''
    define the size of neural network layer
    :param X: input dataset of shape(input size, number of samples)
    :return:
    n_x -- the size of input layer
    n_h -- the size of hidden layer
    n_y -- the size of output layer
    '''
    n_x = X.shape[0]
    n_h = 5  # set the hidden number
    n_y = 1
    return n_x, n_h, n_y


# initialize the model's parameters
def initialize_parameters(n_x, n_h, n_y, seed):
    '''
    randomly initialize the parameters of w and b.
    :param n_x: size of input layer
    :param n_h: size of hidden layer
    :param n_y: size of output layer
    :return: params -- Python dictionary containing our parameters:
                            W1 -- weight matrix of shape (n_h, n_x)
                            b1 -- bias vector of shape (n_h, 1)
                            W2 -- weight matrix of shape (n_y, n_h)
                            b2 -- bias vector of shape(n_y, 1)
    '''
    np.random.seed(seed)  # set random seed for the convenience of reproduce

    W1 = np.random.randn(n_h, n_x).astype(dtype='float32')
    b1 = np.zeros((n_h, 1)).astype(dtype='float32')
    W2 = np.random.randn(n_y, n_h).astype(dtype='float32')
    b2 = np.zeros((n_y, 1)).astype(dtype='float32')

    # using assert statement to make sure our parameters are in the correct shape
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters


# Implement the forward_propagation function:

def forward_propagation(X, parameters):
    '''
    implement forward propagation of neural network
    :param X: input data of shape (n_x, m)
    :param parameters: a python dict including our parameters
    :return:
            A2 -- The sigmoid output of the second activation
            cache -- a dictionary containing 'Z1', 'A1', 'Z2' and 'A2'
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Z1 = np.dot(W1, X) + b1
    Z1 = blas.sgemm(alpha=1.0, a=W1, b=X) + b1
    A1 = np.tanh(Z1)
    # Z2 = np.dot(W2, A1) + b2
    Z2 = blas.sgemm(alpha=1.0, a=W2, b=A1) + b2
    A2 = sigmoid(Z2)

    assert A2.shape == (1, X.shape[1])

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache


# Implement cost function
def compute_cost(A2, Y, parameters):
    '''
    Compute the cross-entropy cost
    :param A2: The sigmoid output of the second activation, of shape (1, m)
    :param Y: 'true' labels vector of shape (1, m)
    :param parameters: Python dict containing our parameters
    :return:
            cost -- cross-entropy cost given equation
    '''
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    assert isinstance(cost, float)

    return cost


# Implement Backward propagation
def backward_propagation(parameters, cache, X, Y):
    '''
    Implement the backward propagation
    :param parameters: python dict containing our parameters
    :param cache: python dict containing 'Z1', 'A1', 'Z2', 'A2' we got above
    :param X: input data of shape(2, number of examples)
    :param Y: 'true' labels vector of shape (1, number of examples)
    :return:
            grads -- python dict including our gradients with respect to different parameters.
    '''

    m = X.shape[1]
    # W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    # dW2 = blas.sgemm(alpha=1.0, a=dZ2, b=A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    # dZ1 = blas.sgemm(alpha=1.0, a=W2.T, b=dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    # dW1 = blas.sgemm(alpha=1.0, a=dZ1, b=X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# using gradient decent to update our parameters

def update_parameters(parameters, grads, learning_rate):
    '''
    Updates parameters using the gradient descent
    :param parameters: python dictionary containing our parameters
    :param grads: python dictionary containing our gradients
    :param learning_rate:  learning rate of gradient decent
    :return:
            parameters -- python dict including our updated parameters
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False, seed=16, alpha=1.1):
    '''
    our neural network model, combing all the function above together
    :param X: training samples
    :param Y: labels
    :param n_h: size of hidden layer
    :param num_iterations: Number of iterations in gradient descent loop
    :param print_cost: if True, print the cost every 1000 iterations
    :return:
            parameters -- parameters learnt by the model
    '''
    if num_iterations is None:
        num_iterations = 1000
    if seed is None:
        seed = 16
    if alpha is None:
        alpha = 1.1

    n_x = layer_sizes(X)[0]
    n_y = layer_sizes(X)[2]

    params = initialize_parameters(n_x, n_h, n_y, seed=seed)

    # Loop gradient descent
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, params)
        cost = compute_cost(A2, Y, params)
        grads = backward_propagation(params, cache, X, Y)
        params = update_parameters(params, grads, learning_rate=alpha)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return params


def predict(parameters, X):
    '''
    Using the learned parameters, predicts a class for each sample in X
    :param parameters: python dict containing our parameters
    :param X: input data
    :return:
        predictions -- vector of predictions of our model (0:sushi, 1:sandwich)
    '''
    A2, cache = forward_propagation(X, parameters)
    predictions = [1 if i > 0.5 else 0 for i in np.squeeze(A2)]
    predictions = np.array(predictions)

    return predictions


def main(argv):
    x, y = upload_img_data()
    X, Y = processing_image_matrix(x, y)

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

    parameters = nn_model(X, Y, 5, num_iterations=num_iterations, print_cost=True, seed=seed, alpha=alpha)
    predictions = predict(parameters, X)

    print("predictions mean = " + str(np.mean(predictions)))
    print('Accuracy: %d' % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')


if __name__ == '__main__':
    main(sys.argv[1:])
