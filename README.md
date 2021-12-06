## Detecting CNN-Generated Images [[Project Page]](https://peterwang512.github.io/CNNDetection/)

**CNN-generated images are surprisingly easy to spot...for now**  
[Sheng-Yu Wang](https://peterwang512.github.io/), [Oliver Wang](http://www.oliverwang.info/), [Richard Zhang](https://richzhang.github.io/), [Andrew Owens](http://andrewowens.com/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/).
<br>In [CVPR](https://arxiv.org/abs/1912.11035), 2020.

<img src='https://peterwang512.github.io/CNNDetection/images/teaser.png' width=1200>

This repository contains models, evaluation code, and training code on datasets from our paper. **If you would like to run our pretrained model on your image/dataset see [(2) Quick start](https://github.com/PeterWang512/CNNDetection#2-quick-start).**

**Jun 20th 2020 Update** Training code and dataset released; test results on uncropped images added (recommended for best performance).

**Oct 26th 2020 Update** Some reported the download link for training data does not work. If this happens, please try the updated alternative links: [1](https://drive.google.com/drive/u/2/folders/14E_R19lqIE9JgotGz09fLPQ4NVqlYbVc) and [2](https://cmu.app.box.com/folder/124997172518?s=4syr4womrggfin0tsfhxohaec5dh6n48)

**Oct 18th 2021 Update** Our method gets 92% AUC on the recently released StyleGAN3 model! For more details, please visit this [link](https://github.com/NVlabs/stylegan3-detector). 

## (1) Setup

### Install packages
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

## (2) Dataset

### Download the dataset
A script for downloading the dataset is as follows: 
```

```

**If the script doesn't work, an alternative will be to download the zip files manually from the above google drive links. One can place the testset, training, and validation set zip files in `dataset/test`, `dataset/train`, and `dataset/val` folders, respectively, and then unzip the zip files to set everything up.**

## (3) Train your models
We provide two example scripts to train our `Blur+JPEG(0.5)` and `Blur+JPEG(0.1)` models. We use `checkpoints/[model_name]/model_epoch_best.pth` as our final model.
```

# Train No Aug
python train.py --name no_aug --blur_prob 0.0 --blur_sig 0.0,3.0 --jpg_prob 0.0 --jpg_method cv2,pil --jpg_qual 100 --dataroot DATAPATH


# Train Blur+JPEG(0.5)
python train.py --name blur_jpg_prob0.5 --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot DATAPATH


# Train Blur+JPEG(0.1)
python train.py --name blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot DATAPATH
```

## (4) Evaluation

After the testset and the model weights are downloaded, one can evaluate the models by running:

```
# Run evaluation script. Model weights need to be downloaded.

```

## (5) Generating Heatmaps

```
python cam.py PATH/TO/TESTDATA {real or fake} --save PATH/TO/SAVE
```

Besides print-outs, the results will also be stored in a csv file in the `results` folder. Configurations such as the path of the dataset, model weight are in `eval_config.py`, and one can modify the evaluation by changing the configurations.


**6/13/2020 Update** Additionally, we tested on uncropped images, and observed better performances on most categories. To evaluate without center-cropping:
```
# Run evaluation script without cropping. Model weights need to be downloaded.
python eval.py --no_crop --batch_size 1
```

The following are the models' performances on the released set, with cropping to 224x224 (as in the paper), and without cropping.

<b>[Blur+JPEG(0.5)]</b>

|Testset   |  Acc (224)  |  AP (224)  |  Recall  |
|:--------:|:------:|:----:|:------:|:----:|
|StyleGAN2 | 97.78% |  99.79%	| 98.85% |
|StyleGAN3    | 99.55%	| 99.98% | 99.25% |

<b>[Blur+JPEG(0.1)]</b>

|Testset   |  Acc (224)  |  AP (224)  |  Recall  |
|:--------:|:------:|:----:|:------:|:----:|
|StyleGAN2 | 98.775%	| 99.99% | 97.65% |
|StyleGAN3    | 98.58%	|99.995%|  99.90% |

## (A) Acknowledgments

This repository borrows from the (https://github.com/PeterWang512/CNNDetection)
