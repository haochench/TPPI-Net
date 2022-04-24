import torch.nn.parallel
from TPPI.loaders.auxil import *
from os.path import join as pjoin
import numpy as np


def get_testLoader(cfg):
    fix_data_path = 'dataset/split_dataset/'
    x_test = np.load(pjoin(fix_data_path + "x_test.npy"))
    y_test = np.load(pjoin(fix_data_path + "y_test.npy"))
    numberofclass = len(np.unique(y_test))
    bands = x_test.shape[-1]
    print("number of class is:{}".format(numberofclass))
    print("bands is:{}".format(bands))
    print("load fix_dataset over")

    test_hyper = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"), y_test))

    kwargs = {'num_workers': 1, 'pin_memory': False}
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=cfg["test"]["batch_size"], shuffle=True, **kwargs)
    print("Successfully created dataloader!")
    return test_loader, numberofclass, bands
