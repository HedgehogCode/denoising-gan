# Denoising GANs

This repository contains code to train denoising GANs with TensorFlow 2.

## Models

* `dcnn_0.05`: Simple DnCNN
* `drcnn_0.02`
* `drcnn_0.05`: Simple DnCNN with internal residuals
* `drcnn_0.10`
* `drcnn_0.20`
* `drcnn-deep_0.05`
* `dunet_0.05`: U-Net
* `dunet+_0.05`: U-Net trained on multiple datasets
* `dunet+_0.0-0.2`: U-Net trained on multiple datasets and noise levels
* `drunet+_0.05`
* `drunet+_0.0-0.2`: DRUNet trained on multiple datasets and noise levels
* **`drugan+_0.0-0.2`: DRUGAN**
* `drugan+-lambda-zero_0.0-0.2`: DRUGAN without adverserial loss
* `drugan+-nora2_0.0-0.2`: DRUGAN without relativistic discriminator
* `drugan+-nora1_0.0-0.2`


## Getting Started

The easiest way to get started is to use VSCode and open the folder in a Remote-Container.

Otherwise, install the dependencies
```
tensorflow==2.4.1
tensorflow-datasets
tensorflow-probability
image-similarity-measures==0.3.5
git+https://github.com/HedgehogCode/tensorflow-datasets-bw.git@0.6.4
```

## Usage

Use the following environment variables to control the scripts.
* `DNGAN_DEBUG`: Debug mode, only train for a few steps
* `DNGAN_LOGS_PREFIX`: Folder for the Tensorboard logs
* `DNGAN_CHECKPOINTS_PREFIX`: Folder for the checkpoints
* `DNGAN_CONFIG`: Path to the config json file (See [configs/](configs/))

Training the generator with MSE loss:
```
$ python train_dae.py
```

Training the denoising GAN with all losses (based on a pretrained generator):
```
$ python train_dngan.py
```

Exporting the generator to a h5 file:
```
$ python checkpoint_to_h5.py -c /path/to/checkpoints configs/my_config.json /path/to/model.h5
```

## References

**DnCNN:** K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, [“Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising,”](https://doi.org/10.1109/TIP.2017.2662206) IEEE Trans. on Image Process., vol. 26, no. 7, pp. 3142–3155, Jul. 2017, doi: 10.1109/TIP.2017.2662206.

**DRUNet** K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, and R. Timofte, [“Plug-and-Play Image Restoration with Deep Denoiser Prior,”](http://arxiv.org/abs/2008.13751) arXiv:2008.13751 [cs, eess], Aug. 2020


## TODO

* Move evaluate.py script to other repository, also adapt the dependencies
* Add download links for trained models
* Add more references
