# Introduction

Source code to reproduce the results in “JPEG Steganalysis Based on ResNeXt with Gauss Partial Derivative Filters” 


This code makes the necessary modifications on the [code](https://github.com/GuanshuoXu/caffe_deep_learning_for_steganalysis) provided by Dr. Xu.
# Building Instructions

The DCT kernels are saved in /kernels and the GPD kernels are saved in /include/caffe. The directory to access them are hard-coded in /include/caffe/filler.hpp. So before building, please change the directories to make sure the DCT kernels and GPD kernels can be found.

The code will compile with cudnnv6. If you are using cudnnv5, see the instruction in /cudnn_hpp_version

# Features
This code has following features compared with the official Caffe.

1) Memory-efficient BN-ReLU combo (bn_conv and relu_recover). Please see bn_conv_layer and relu_recover_layer.
2) More stable testing performance (important for running average based BN) by parameter-wise averaging across N training iterations before testing. Please see the Step(int iters) function in solver.cpp. Usage: set use_polyak to true and  num_iter_polyak in solver.prototxt.
3) image_data_steganalysis_jpeg_dct_layer: a new input layer for jpeg_steganalysis that read jpeg images from hard drive and output BDCT coefficients. This layer is able to do per-epoch random shuffling and syncronized random mirroring and rotation for each cover-stego pair. This layer requires cover and its corresponding stego to have the save file name. Please refer to image_data_steganalysis_jpeg_dct_layer.cpp for more details.
4) bdct_to_spatial_layer to tranform BDCT coeffients to spatial domain.
5) quant_trunc_abs_layer to perform element-wise quantization, trunction and absolute operations.

# Examples
There is a example provided in examples/jpeg_steganalysis for QF75. Minumum required GPU memory is 15GB. Recommend P100.

1) Change the Caffe dir in cmd.sh and cmd_test.sh. The cmd_test.sh is used to output probabilities only.
2) Set the source, cover_dir, and stego_dir in the input layer (in RXGNet.prototxt).
3) The source is a txt file, each line contain a number (from 1 ~ 10000 for BOSSBase). See the txt files in /rand_num_generators. cover_dir and stego_dir simply contain images in '.jpg' format.
