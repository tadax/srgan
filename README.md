# Super Resolution using Generative Adversarial Network (SRGAN)

This is an implementation of the SRGAN model proposed in the paper
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](
https://arxiv.org/abs/1609.04802)
with TensorFlow.

# Requirements

- Python 3
- TensorFlow
- OpenCV
- dlib

# Usage

## I. Pretrain the VGG-19 model.

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

The pretrained VGG-19 model will be stored in "vgg19/model".


## II. Train the SRGAN (ResNet-Generator and Discriminator) model.

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

The result will be stored in "src/result".


# Results

## LFW

After 20 epochs

![result1](results/000000001.jpg)

![result2](results/000000002.jpg)

![result3](results/000000003.jpg)

![result4](results/000000004.jpg)

![result5](results/000000005.jpg)

![result6](results/000000006.jpg)

![result7](results/000000007.jpg)

![result8](results/000000008.jpg)

![result9](results/000000009.jpg)

![result10](results/000000010.jpg)

![result11](results/000000011.jpg)

![result12](results/000000012.jpg)

![result13](results/000000013.jpg)

![result14](results/000000014.jpg)

![result15](results/000000015.jpg)

![result16](results/000000016.jpg)

![result17](results/000000017.jpg)

![result18](results/000000018.jpg)

![result19](results/000000019.jpg)

![result20](results/000000020.jpg)

![result21](results/000000021.jpg)

![result22](results/000000022.jpg)

![result23](results/000000023.jpg)

![result24](results/000000024.jpg)

![result25](results/000000025.jpg)

![result26](results/000000026.jpg)

![result27](results/000000027.jpg)

![result28](results/000000028.jpg)

![result29](results/000000029.jpg)

![result30](results/000000030.jpg)

![result31](results/000000031.jpg)

![result32](results/000000032.jpg)


# Appendix

## Adversarial loss 

This implementation adopts the least squares loss function instead 
of the sigmoid cross entropy loss function for the discriminator.

See the details: [Least Squares Generative Adversarial Networks](
https://arxiv.org/abs/1611.04076)

