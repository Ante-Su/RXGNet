#ifndef CAFFE_QUANT_TRUNC_ABS_LAYER_HPP_
#define CAFFE_QUANT_TRUNC_ABS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template<typename Dtype>
class QuantTruncAbsLayer: public NeuronLayer<Dtype> {
public:
	explicit QuantTruncAbsLayer(const LayerParameter& param) :
			NeuronLayer<Dtype>(param) {
	}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "QuantTruncAbs";
	}

	virtual inline int ExactNumBottomBlobs() const {
		return 1;
	}
	virtual inline int ExactNumTopBlobs() const {
		return 1;
	}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		NOT_IMPLEMENTED;
	}
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}

	int threshold_;
};

} //namespace caffe

#endif // CAFFE_QUANT_TRUNC_ABS_LAYER_HPP_
