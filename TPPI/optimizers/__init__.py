import logging
from torch.optim import SGD, Adam

logger = logging.getLogger("HSIC")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
}


def get_optimizer(cfg):
    if cfg["train"]["optimizer"] is None:
        logger.info("Using SGD optimizer")
        return SGD

    else:
        opt_name = cfg["train"]["optimizer"]["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        logger.info("Using {} optimizer".format(opt_name))
        return key2opt[opt_name]
