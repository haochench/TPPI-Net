"""
Evaluating inference time (whole HSI) and classification accuracy (test set) of TPPP-Nets
"""
import os
import torch
import argparse
import numpy as np
import yaml
import scipy.io as sio
import time
import auxil
from TPPI.utils import convert_state_dict
from TPPI.models import get_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def predict_patches(data, model, cfg, device):
    transfer_data_start = time.time()
    data = data.to(device)
    transfer_data_end = time.time()
    transfer_time = transfer_data_end - transfer_data_start
    predicted = []
    bs = cfg["prediction"]["batch_size"]
    tsp = time.time()
    with torch.no_grad():
        for i in range(0, data.shape[0], bs):
            end_index = i + bs
            batch_data = data[i:end_index]
            outputs = model(batch_data)
            [predicted.append(a) for a in outputs.data.cpu().numpy()]
    tep = time.time()
    prediction_time = tep - tsp
    return prediction_time, transfer_time, np.array(predicted)


def timeCost_TPPP(cfg, logdir):
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

    # image processing
    time_pre_start = time.time()
    # StandardScaler
    shapeor = img.shape
    img = img.reshape(-1, img.shape[-1])
    img = StandardScaler().fit_transform(img)
    img = img.reshape(shapeor)
    # create patch
    data = auxil.creat_PP(cfg["data"]["PPsize"], img)
    # NHWC -> NCHW
    data = data.transpose(0, 3, 1, 2)
    data = torch.from_numpy(data).float()
    time_pre_end = time.time()
    time_pre_processing = time_pre_end - time_pre_start
    print("creat patch {} data over!", data.shape)

    # setup model:
    model = get_model(cfg['model'], cfg['data']['dataset'])
    state = convert_state_dict(
        torch.load(os.path.join(logdir, cfg["train"]["best_model_path"]))[
            "model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # transfer model to GPU
    ts1 = time.time()
    model.to(device)
    ts2 = time.time()

    # predicting
    print("predicting...")
    pt, tt, outputs = predict_patches(data, model, cfg, device)
    print(outputs.shape)

    # get result and reshape
    comb_s = time.time()
    outputs = np.array(outputs)
    pred = np.argmax(outputs, axis=1)
    if cfg['data']['dataset'] == 'IP':
        pred = np.reshape(pred, (145, 145))
    elif cfg['data']['dataset'] == 'PU':
        pred = np.reshape(pred, (610, 340))
    elif cfg['data']['dataset'] == 'SV':
        pred = np.reshape(pred, (512, 217))
    elif cfg['data']['dataset'] == 'KSC':
        pred = np.reshape(pred, (512, 614))
    comb_e = time.time()

    # show predicted result
    pred += 1
    auxil.decode_segmap(pred)

    # computing classification accuracy
    pred = pred[position == 1]
    gt = gt[position == 1]
    classification, confusion, result = auxil.reports(pred, gt)
    result_info = "OA AA Kappa and each Acc:\n" + str(result)

    # report time cost and accuracy
    print("******************** Time ***********************")
    print("Pre_processing time is:", time_pre_processing)
    print("Transfer time is:", tt + (ts2-ts1), "  model:", ts2 - ts1, "  data:", tt)
    print("Prediction time is:", pt)
    print("combine time is:", comb_e-comb_s)
    print('Total inference time is:', time_pre_processing + tt + (ts2-ts1) +pt +comb_e-comb_s)

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
    timeCost_TPPP(cfg, logdir)

