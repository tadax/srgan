#!/bin/sh

# CIFAR-100
wget -O 'cifar-100-python.tar.gz' 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
tar -xvzf cifar-100-python.tar.gz
mv cifar-100-python raw
rm cifar-100-python.tar.gz

