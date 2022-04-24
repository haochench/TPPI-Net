import os
import shutil
import scipy.io as sio
import yaml
import numpy as np
import random
import argparse
from os.path import join as pjoin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def setDir():
    filepath = 'dataset/split_dataset'
    if not os.path.isdir(filepath):
        os.makedirs(filepath)


def random_unison(a, b, c, rstate=None):
    assert len(a) == len(b) & len(a) == len(c)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p], c[p]


# load data , PCA (optional) and Normalization
def loadData(cfg):
    data_path = 'dataset/'
    dataset = cfg['data']["dataset"]
    num_components = cfg['data']['num_components']
    if dataset == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
    elif dataset == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
    elif dataset == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif dataset == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    else:
        print("NO DATASET")
        exit()
    print("load {} original image successfully".format(dataset))

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])

    # PCA or not
    if num_components is not None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components

    # Normalization
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels))-1
    return data, labels, num_class


def padWithZeros(X, margin):
    """
    :param X: input, shape:[H,W,C]
    :param margin: padding
    :return: new data, shape:[H+2*margin, W+2*margin, C]
    """
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def get_patch(cfg, data, x, y):
    """
    get one patch according it's position
    """
    windowSize = cfg['data']['PPsize']
    margin = int((windowSize - 1) / 2)
    x += margin
    y += margin
    zeroPaddeddata = padWithZeros(data, margin=margin)
    patch = zeroPaddeddata[x - margin:x + margin + 1, y - margin:y + margin + 1]
    return patch


def creat_PP(cfg, X, y):
    windowSize = cfg['data']['PPsize']
    removeZeroLabels = cfg['data']["remove_zeros"]
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchesLocations = []

    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchesLocations.append([r-margin, c-margin])
            patchIndex = patchIndex + 1
    # remove unlabeled patches
    patchesLocations = np.asarray(patchesLocations)
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLocations = patchesLocations[patchesLabels > 0]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels.astype("int"), patchesLocations


# splitting dataset
def split_data(pixels, labels, indexes, percent, rand_state=69):
        pixels_number = np.unique(labels, return_counts=1)[1]
        train_set_size = [int(np.ceil(a*percent)) for a in pixels_number]
        tr_size = int(sum(train_set_size))
        te_size = int(sum(pixels_number)) - int(sum(train_set_size))
        sizetr = np.array([tr_size]+list(pixels.shape)[1:])
        sizete = np.array([te_size]+list(pixels.shape)[1:])
        tr_index = []
        te_index = []
        train_x = np.empty((sizetr))
        train_y = np.empty((tr_size), dtype=int)
        test_x = np.empty((sizete))
        test_y = np.empty((te_size),dtype=int)
        trcont = 0
        tecont = 0
        for cl in np.unique(labels):
            pixels_cl = pixels[labels == cl]
            labels_cl = labels[labels == cl]
            indexes_cl = indexes[labels == cl]
            pixels_cl, labels_cl, indexes_cl = random_unison(pixels_cl, labels_cl, indexes_cl, rstate=rand_state)
            for cont, (a, b, c) in enumerate(zip(pixels_cl, labels_cl, indexes_cl)):
                if cont < train_set_size[cl]:
                    train_x[trcont, :, :, :] = a
                    train_y[trcont] = b
                    tr_index.append(c)
                    trcont += 1
                else:
                    test_x[tecont, :, :, :] = a
                    test_y[tecont] = b
                    te_index.append(c)
                    tecont += 1
        tr_index = np.asarray(tr_index)
        te_index = np.asarray(te_index)
        train_x, train_y, tr_index = random_unison(train_x, train_y, tr_index, rstate=rand_state)
        return train_x, test_x, train_y, test_y, tr_index, te_index


if __name__ == '__main__':
    with open("configs/config.yml") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # load data
    data, labels, num_class = loadData(cfg)

    # create patches
    newdata, newlabels, indexes = creat_PP(cfg, data, labels)

    # splitting dataset
    x_train, x_test, y_train, y_test, train_index, test_index = split_data(newdata, newlabels, indexes, cfg['data']["tr_percent"], cfg['data']["rand_state"])
    x_val, x_test, y_val, y_test, val_index, new_test_index = split_data(x_test, y_test, test_index, cfg['data']["val_percent"], cfg['data']["rand_state"])
    del newdata, newlabels

    # positions of testSet
    test_positions = np.zeros(labels.shape)
    for pos in new_test_index:
        test_positions[pos[0]][pos[1]] = 1

    # show the shape of each dataset
    # print("x_train shape:", x_train.shape)
    # print("x_val shape:", x_val.shape)
    # print("x_test shape:", x_test.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_val shape:", y_val.shape)
    # print("y_test shape:", y_test.shape)

    setDir()
    fix_data_path = 'dataset/split_dataset/'

    # save each dataset and testSet position；
    np.save(pjoin(fix_data_path+"testSet_position.npy"), test_positions)
    np.save(pjoin(fix_data_path+"x_train.npy"), x_train)
    np.save(pjoin(fix_data_path+"x_val.npy"), x_val)
    np.save(pjoin(fix_data_path+"x_test.npy"), x_test)
    np.save(pjoin(fix_data_path + "y_train.npy"), y_train)
    np.save(pjoin(fix_data_path + "y_val.npy"), y_val)
    np.save(pjoin(fix_data_path + "y_test.npy"), y_test)
    print("creat dataset over!")

