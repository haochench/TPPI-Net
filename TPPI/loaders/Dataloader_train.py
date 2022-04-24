import torch.nn.parallel
from TPPI.loaders.auxil import *
from os.path import join as pjoin
import numpy as np


def get_trainLoader(cfg):
    fix_data_path = 'dataset/split_dataset/'
    x_train = np.load(pjoin(fix_data_path + "x_train.npy"))
    y_train = np.load(pjoin(fix_data_path + "y_train.npy"))
    x_val = np.load(pjoin(fix_data_path + "x_val.npy"))
    y_val = np.load(pjoin(fix_data_path + "y_val.npy"))
    numberofclass = len(np.unique(y_train))
    bands = x_train.shape[-1]
    print("number of class is:{}".format(numberofclass))
    print("bands is:{}".format(bands))
    print("load fix_dataset over")

    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"), y_train))
    val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"), y_val))

    kwargs = {'num_workers': 1, 'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=cfg["train"]["batch_size"], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=cfg["test"]["batch_size"], shuffle=False, **kwargs)
    print("Successfully created dataloader!")
    return train_loader, val_loader, numberofclass, bands
