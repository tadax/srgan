#!/bin/sh

if [ -e raw ]; then
    echo 'Already exists.'
    exit
fi

wget -O 'lfw.tgz' 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
tar -xvzf lfw.tgz
mv lfw raw
rm lfw.tgz

