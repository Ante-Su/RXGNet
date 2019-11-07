#!/usr/bin/env sh

/home/user/suante/Caffe_RXGNet/build/tools/caffe train --solver=solver.prototxt --gpu 1 2>&1 | tee model_train.log
