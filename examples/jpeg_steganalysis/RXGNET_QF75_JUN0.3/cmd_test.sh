#!/usr/bin/env sh

/home/user/suante/Caffe_RXGNet/build/tools/caffe test -model RXGNet_test.prototxt -weights inference90000.caffemodel -gpu 0 -iterations 200 -prob prob90000.txt
