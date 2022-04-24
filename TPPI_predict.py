"""
Evaluating inference time (whole HSI) and classification accuracy (test set) of TPPI-Nets
"""
import os
import torch
import argparse
import numpy as np
import yaml
import scipy.io as sio
import time
import auxil
from TPPI.models import get_model
from TPPI.utils import convert_state_dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# @torchsnooper.snoop()
def test(cfg, logdir):
    name = cfg["data"]["dataset"]
    device = auxil.get_device()

    # Setup image
    teposition_path = 'dataset/split_dataset/testSet_position.npy'
    position = np.load(teposition_path)
    org_img_path = 'dataset/'
    if name == "IP":
        img = sio.loadmat(os.path.join(org_img_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        gt = sio.loadmat(os.path.join(org_img_path, 'indian_pines_gt.mat'))['indian_pines_gt']
    elif name == "PU":
        img = sio.loadmat(os.path.join(org_img_path, 'paviaU.mat'))['paviaU']
        gt = sio.loadmat(os.path.join(org_img_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name == "SV":
        img = sio.loadmat(os.path.join(org_img_path, 'salinas_corrected.mat'))['salinas_corrected']
        gt = sio.loadmat(os.path.join(org_img_path, 'salinas_gt.mat'))['salinas_gt']
    else:
        print("No this dataset")
    print("data shape:", img.shape)
    print("GT shape:", gt.shape)

    time_pre_start = time.time()
    # StandardScaler
    shapeor = img.shape
    img = img.reshape(-1, img.shape[-1])
    img = StandardScaler().fit_transform(img)
    img = img.reshape(shapeor)

    # split image if your GPU or CPU don't have enough RAM
    # if name == 'IP':
    #     p = 40
    #     img = img[:p, :p, :]
    #     gt = gt[:p, :p]
    #     test_position = test_position[:p, :p]
    # elif name == "PU":
    #     p = 60
    #     img = img[:p, :p, :]
    #     gt = gt[:p, :p]
    #     test_position = test_position[:p, :p]
    # elif name == "SV":
    #     p = 40
    #     img = img[:p, :p, :]
    #     gt = gt[:p, :p]
    #     test_position = test_position[:p, :p]

    # padding
    Margin = (cfg["data"]["PPsize"] - 1) // 2
    img = auxil.padWithZeros(img, margin=Margin)
    img = img.astype("float32")

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    time_pre_end = time.time()
    time_pre_processing = time_pre_end - time_pre_start

    # Setup Model
    model = get_model(cfg['model'], cfg['data']['dataset'])
    state = convert_state_dict(
        torch.load(os.path.join(logdir, cfg["train"]["best_model_path"]))[
            "model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # transfer model and data to GPU
    ts1 = time.time()
    model.to(device)
    ts2 = time.time()
    images = img.to(device)
    ts3 = time.time()

    # predict time
    start_ts = time.time()
    outputs = model(images)

    # the predicted result
    pred = outputs.data.max(0)[1].cpu().numpy()  # (145,145)
    end_ts = time.time()

    # show predicted result
    pred += 1
    print('pred shape', pred.shape)
    auxil.decode_segmap(pred)

    # computing classification accuracy
    pred = pred[position == 1]
    gt = gt[position == 1]
    classification, confusion, result = auxil.reports(pred, gt)
    result_info = "OA AA Kappa and each Acc: \n" + str(result)

    # report inference time
    print("******************** Time ***********************")
    print("Pre_processing time is:", time_pre_processing)
    print("Transfer time is:", ts3 - ts1, "  model:", ts2 - ts1, "  data:", ts3 - ts2)
    print("Prediction time is:", (end_ts - start_ts))
    print('Total inference time is:', time_pre_processing+ts3-ts1+end_ts-start_ts)

    # report classification accuracy
    print("****************** Accuracy *********************")
    print(result_info)
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    logdir = os.path.join("runs", cfg["model"], str(cfg["run_ID"]))
    test(cfg, logdir)

