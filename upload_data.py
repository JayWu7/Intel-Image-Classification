import numpy as np
import matplotlib.pyplot as plt
from os import listdir


def upload_img_data(directory='./dataset/train/data'):
    '''
    read the images and return the matrix of these images
    :param directory: directory which store these images
    :return:  numpy arrays with shape(m, 300,300,3)
    '''
    images_name = [n for n in listdir(directory) if not n.startswith('.')]
    images = []
    labels = []
    for na in images_name:
        im = plt.imread('{}/{}'.format(directory, na), 'JPG')
        images.append(im)
        if na.startswith('sushi'):  # 0 means this pic is a sushi pic, 1 means sandwich
            labels.append(0)
        else:
            labels.append(1)
    images = np.array(images, dtype=np.float32)  # transfer the list type to numpy ndarray
    return images, labels


def processing_image_matrix(images, labels):
    '''
    using reduction and normalization to process our data.
    :param images: numpy array, store the matrix of images
    :return: numpy array, with shape(300*300*3, m)
    '''
    images_flatten = images.reshape(images.shape[0], -1).T
    # normalize data
    train_set_x = images_flatten / 255
    train_set_y = np.array(labels, dtype=np.float32).reshape((1, len(labels)))
    return train_set_x, train_set_y