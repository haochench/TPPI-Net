from TPPI.models.TPPP import CNN_1D, CNN_2D_new, CNN_3D_new, HybridSN, SSAN, pResNet, SSRN
from TPPI.models.TPPI import CNN_1D_TPPI, CNN_2D_TPPI_new, CNN_3D_TPPI_new, HybridSN_TPPI, SSAN_TPPI,pResNet_TPPI, SSRN_TPPI


def get_model(modelName, dataset):
    model = get_model_instance(modelName)
    model = model(dataset)
    return model


def get_model_instance(name):
    try:
        return {
            'CNN_1D': CNN_1D,
            'CNN_1D_TPPI': CNN_1D_TPPI,
            'CNN_2D': CNN_2D_new,
            'CNN_2D_TPPI': CNN_2D_TPPI_new,
            'CNN_3D': CNN_3D_new,
            'CNN_3D_TPPI': CNN_3D_TPPI_new,
            'HybridSN': HybridSN,
            'HybridSN_TPPI': HybridSN_TPPI,
            'SSRN': SSRN,
            'SSRN_TPPI': SSRN_TPPI,
            'pResNet': pResNet,
            'pResNet_TPPI': pResNet_TPPI,
            'SSAN': SSAN,
            'SSAN_TPPI': SSAN_TPPI,
        }[name]
    except:
        raise ("Model {} not available".format(name))


