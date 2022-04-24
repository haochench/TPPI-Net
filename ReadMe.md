# PyTorch-Hyperspectral image classification
## Introduction
This repository is a PyTorch implementation for hyperspectral image classification.

**Official implementation of TPPI-Net** and **Unofficial implementation of Others.**

**Contact**: chenhao1@stu.scu.edu.cn  or  994161476@qq.com

If you find this code useful in your research, please consider citing:
'''

'''

**Implemented networks including**:
1. TPPI-Nets: 
   1. paper: A Novel Network Framework and Model for Efficient Hyperspectral Image Classification
2. TPPP-Nets:
   1. 1D CNN:
      1. paper: Chen, Y., Jiang, H., Li, C., Jia, X., & Ghamisi, P. (2016). Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing, 54(10), 6232–6251. https://doi.org/10.1109/TGRS.2016.2584107
      2. source code: I can't find it -.- 
   2. 2D CNN:
      1. paper1: Yang, X., Ye, Y., Li, X., Lau, R. Y. K., Zhang, X., & Huang, X. (2018). Hyperspectral image classification with deep learning models. IEEE Transactions on Geoscience and Remote Sensing, 56(9), 5408–5423. https://doi.org/10.1109/TGRS.2018.2815613
      2. source code of paper1: -.-
      3. paper2: Chen, Y., Jiang, H., Li, C., Jia, X., & Ghamisi, P. (2016). Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing, 54(10), 6232–6251. https://doi.org/10.1109/TGRS.2016.2584107
      4. source code of paper2: -.-
   3. 3D CNN:
      1. paper1: Ben Hamida, A., Benoit, A., Lambert, P., & Ben Amar, C. (2018). 3-D deep learning approach for remote sensing image classification. IEEE Transactions on Geoscience and Remote Sensing, 56(8), 4420–4434. https://doi.org/10.1109/TGRS.2018.2818945
      2. source code of paper1: https://github.com/AminaBh/3D_deepLearning_for_hyperspectral_images
      3. paper2: Chen, Y., Jiang, H., Li, C., Jia, X., & Ghamisi, P. (2016). Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing, 54(10), 6232–6251. https://doi.org/10.1109/TGRS.2016.2584107
      4. source code of paper2: -.-
   4. HybridSN: 
      1. paper: Roy, S. K., Krishna, G., Dubey, S. R., & Chaudhuri, B. B. (2020). HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification. IEEE Geoscience and Remote Sensing Letters, 17(2), 277–281. https://doi.org/10.1109/LGRS.2019.2918719
      2. source code: https://github.com/gokriznastic/HybridSN
   5. SSAN: 
      1. paper: Sun, H., Zheng, X., Lu, X., & Wu, S. (2020). Spectral-Spatial Attention Network for Hyperspectral Image Classification. IEEE Transactions on Geoscience and Remote Sensing, 58(5), 3232–3245. https://doi.org/10.1109/TGRS.2019.2951160
      2. source code: I got the implementation of SSAN by contacting Prof. Zheng: xiangtaoz@gmail.com 
   6. pResNet:
      1. paper: Paoletti, M. E., Haut, J. M., Fernandez-Beltran, R., Plaza, J., Plaza, A. J., & Pla, F. (2019). Deep pyramidal residual networks for spectral-spatial hyperspectral image classification. IEEE Transactions on Geoscience and Remote Sensing, 57(2), 740–754. https://doi.org/10.1109/TGRS.2018.2860125
      2. source code: https://github.com/mhaut/pResNet-HSI
   7. SSRN:
      1. paper: Zhong, Z., Li, J., Luo, Z., & Chapman, M. (2018). Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework. IEEE Transactions on Geoscience and Remote Sensing, 56(2), 847–858. https://doi.org/10.1109/TGRS.2017.2755542
      2. source code: https://github.com/zilongzhong/SSRN
## Usage
### Setup config file
```
Modify ‘configs/config.yml’
```
### Data set preparation

```
python create_dataset.py
```

### Training

```
python train.py
```

### Evaluation
For TPPI-Nets, run: 
```
python TPPI_predict.py
```
For TPPP-Nets, run: 
```
python TPPP_predict.py
```


