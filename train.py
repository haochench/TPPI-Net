import os
import yaml
import shutil
import random
import argparse
from tensorboardX import SummaryWriter
import torch
import torch.nn.parallel
import numpy as np
import time
from torch.autograd.variable import Variable
from TPPI.models import get_model
from TPPI.optimizers import get_optimizer
from TPPI.schedulers import get_scheduler
from TPPI.loaders.Dataloader_train import get_trainLoader
from TPPI.utils import get_logger
import auxil


def train(cfg, train_loader, val_loader, model, loss_fn, optimizer, device, tr_writer, val_writer, logdir, logger):
    start_epoch = 0
    continue_path = os.path.join(logdir, "continue_model.pkl")
    if os.path.isfile(continue_path):
        logger.info(
            "Loading model and optimizer from checkpoint '{}'".format(continue_path)
        )
        checkpoint = torch.load(continue_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"]
        logger.info(
            "Loaded checkpoint '{}' (iter {})".format(
                continue_path, checkpoint["epoch"]
            )
        )
    else:
        logger.info("No checkpoint found at '{}'".format(continue_path))

    best_acc = -1
    epoch = start_epoch
    flag = True
    while epoch <= cfg["train"]["epochs"] and flag:
        model.train()
        train_accs = np.ones((len(train_loader))) * -1000.0
        train_losses = np.ones((len(train_loader))) * -1000.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            train_losses[batch_idx] = loss.item()
            train_accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss = np.average(train_losses)
        train_acc = np.average(train_accs)
        fmt_str = "Iter [{:d}/{:d}]  \nTrain_loss: {:f}  Train_acc: {:f}"
        print_str = fmt_str.format(
            epoch + 1,
            cfg["train"]["epochs"],
            train_loss,
            train_acc,
        )
        tr_writer.add_scalar("loss", train_loss, epoch+1)
        print(print_str)
        logger.info(print_str)

        state = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }
        # save to the continue path
        torch.save(state, continue_path)

        epoch += 1

        if (epoch + 1) % cfg["train"]["val_interval"] == 0 or (epoch + 1) == cfg["train"]["epochs"]:
            model.eval()
            val_accs = np.ones((len(val_loader))) * -1000.0
            val_losses = np.ones((len(val_loader))) * -1000.0
            with torch.no_grad():
                for batch_idy, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                    outputs = model(inputs)
                    val_losses[batch_idy] = loss_fn(outputs, targets).item()
                    val_accs[batch_idy] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
                val_loss = np.average(val_losses)
                val_acc = np.average(val_accs)

            fmt_str = "Val_loss: {:f}  Val_acc: {:f}"
            print_str = fmt_str.format(
                val_loss,
                val_acc,
            )
            val_writer.add_scalar("loss", val_loss, epoch)
            print(print_str)
            logger.info(print_str)

            if val_acc > best_acc:
                best_acc = val_acc
                state = {
                    'epoch': epoch + 1,
                    'best_acc': best_acc,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }
                torch.save(state, os.path.join(logdir, "best_model.pth.tar"))

        if epoch == cfg["train"]["epochs"]:
            flag = False
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSIC model Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    logdir = os.path.join("runs", cfg["model"], str(cfg["run_ID"]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    tr_writer = SummaryWriter(log_dir=os.path.join(logdir+"/train/"))
    val_writer = SummaryWriter(log_dir=os.path.join(logdir+"/val/"))
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let begin!")

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = auxil.get_device()

    # Setup Dataloader
    train_loader, val_loader, num_classes, n_bands = get_trainLoader(cfg)

    # Setup Model
    model = get_model(cfg['model'], cfg['data']['dataset'])
    model = model.to(device)
    print("model load successfully")

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["train"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    # Setup lr_scheduler
    scheduler = get_scheduler(optimizer, cfg["train"]["lr_schedule"])
    best_err1 = 100

    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # training model
    train(cfg, train_loader, val_loader, model, loss_fn, optimizer, device, tr_writer, val_writer, logdir, logger)

    # training over!
    print("Training is over!")
    logger.info("Training is over!")
