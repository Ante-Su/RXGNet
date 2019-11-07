#ifndef CAFFE_QUANT_TRUN_LAYER_HPP_
#define CAFFE_QUANT_TRUN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

//quantzation and trunction
template <typename Dtype>
class TruncLayer : public NeuronLayer<Dtype> {
 public:
  explicit TruncLayer(const LayerParameter& param)   : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Trunc"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  
//the parameter of quantization
   Dtype quant;
};

}  // namespace caffe

#endif  // CAFFE_QUANT_TRUN_LAYER_HPP_
