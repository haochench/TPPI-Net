* run_ID: name
* GPU: value, GPU to be used
* model: name. [options: 'CNN_1D','CNN_1D_TPPI','CNN_2D','CNN_2D_TPPI','CNN_3D',
            'CNN_3D_TPPI','HybridSN','HybridSN_TPPI','SSRN','SSRN_TPPI','pResNet',
            'pResNet_TPPI','SSAN','SSAN_TPPI',]
* data:
  * dataset: name. [options: 'IP', 'PU', 'SV', 'KSC' ]
  * PPsize: value. The spatial size of patch samples.
  * remove_zeros: removing the unlabeled data. [options: true, false]
  * tr_percent: value. The proportion of the training set in all samples.
  * val_percent: value. The proportion of the validation set in all samples.
  * rand_state: value. Random seed.
* train:
  * epochs: value.  
  * batch_size: value. 
  * val_interval: value.  
  * n_workers: value. 
  * print_interval: value.  
  * optimizer:
    * name: 'sgd' 
    * lr: 0.01 
    * weight_decay: 0.0001 
    * momentum: 0.9 
  * loss:
    * name: 'cross_entropy' 
  * lr_schedule:
  * continue_path: Save path of the model for each epoch. 
  * best_model_path: Save path of the best model.
* test:
  * batch_size: value. 
* prediction:
  * batch_size: Ensure that TPPP-Net can use the same VRAM as TPPI-Net. batch_size=(H+n-1)\*(W+n-1)/(n\*n)
  