# Super Resolution using Generative Adversarial Network (SRGAN)

This is an implementation of the SRGAN model proposed in the paper
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
with TensorFlow.

# Requirements

- TensorFlow
- OpenCV
- dilb

# Usage

## I. Pre-train the VGG-19 model.

Download the dataset with:

```
$ ./vgg19/cifar_100/download.sh
```

Preprocess the dataset with:

```
$ python vgg19/cifar_100/preprocess.py
```

Train with:

```
$ python vgg19/train.py
```

You will get the pretrained VGG-19 model in the directory [vgg19/model/](vgg19/model/).


## II. Train SRGAN (SR ResNet + VGG + Discriminator) model.

Download the dataset with:

```
$ ./srgan/lfw/download.sh
```

Preprocess the dataset with:

```
$ python srgan/lfw/preprocess.py
```

Train with:

```
$ python srgan/train.py
```

The result will be stored in the directory [srgan/result/](srgan/result/).


# Results

## LFW

![result1](sample/1.jpg)

![result2](sample/2.jpg)

![result3](sample/3.jpg)
