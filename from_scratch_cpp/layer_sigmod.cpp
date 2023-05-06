#include "layer_sigmod.h"

double_t sigma(double_t x) {
    return 1 / (1 + exp(-x));
}

double_t der_sigma(double_t x) {
    return sigma(x) * (1 - sigma(x));
}

std::vector<tensor> layer_sigmod::forward(const std::vector<tensor>& ins) {
	_ins = ins;
	std::vector<tensor> outs = ins;
	for (size_t i = 0; i < ins.size(); i++)
		outs[i].apply(tanh);
	return outs;
}

std::vector<tensor> layer_sigmod::backward(const std::vector<tensor>& grads) {
	for (size_t i = 0; i < _ins.size(); i++) {
		_ins[i].apply(der_sigma);
		_ins[i] *= grads[i];
	}
	return _ins;
}
