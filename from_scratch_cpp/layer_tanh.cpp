#include "layer_tanh.h"

double_t der_tanh(double_t x) {
	double_t ch = cosh(x);
	return 1 / (ch * ch);
}

std::vector<tensor> layer_tanh::forward(const std::vector<tensor>& ins) {
	_ins = ins;
	std::vector<tensor> outs = ins;
	for (size_t i = 0; i < ins.size(); i++)
		outs[i].apply(tanh);
	return outs;
}

std::vector<tensor> layer_tanh::backward(const std::vector<tensor>& grads) {
	for (size_t i = 0; i < _ins.size(); i++) {
		_ins[i].apply(der_tanh);
		_ins[i] *= grads[i];
	}
	return _ins;
}
