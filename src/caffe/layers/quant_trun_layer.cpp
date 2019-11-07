#include <algorithm>
#include <vector>

#include "caffe/layers/quant_trun_layer.hpp"

namespace caffe {

template <typename Dtype>
void Quant_TrunLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype quant = this->layer_param_.quant_trun_param().quant();
 // std::cout<<"****************************************************************************************************"<<std::endl;
  for (int i = 0; i < count; ++i) {
    if(bottom_data[i]>0)
		top_data[i] = std::min(((int)((bottom_data[i])/quant)), int(4));
	else
		top_data[i] = std::max(((int)((bottom_data[i])/quant)), int(-4));
   /* if(i%256==0)
	std::cout<<std::endl;
    if(quant==2)
	 std::cout<<top_data[i]<<" ";*/
  }
}

template <typename Dtype>
void Quant_TrunLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
}



#ifdef CPU_ONLY
STUB_GPU(Quant_TrunLayer);
#endif

INSTANTIATE_CLASS(Quant_TrunLayer);
REGISTER_LAYER_CLASS(Quant_Trun);
}  // namespace caffe
