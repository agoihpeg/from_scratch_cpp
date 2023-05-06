#include "layer_linear.h"
#include "linalg.h"
#include <assert.h>

layer_linear::layer_linear(size_t in, size_t out, distribution d)
	: _w(random_tensor({ in, out }, d)), _b(random_tensor({ 1, out }, d)), _ins() {
	_w_grad = zeros_like(_w);
	_b_grad = zeros_like(_b);
	_params = { &_w,&_b };
	_grads = { &_w_grad,&_b_grad };
}

layer_linear::layer_linear(const tensor& w, const tensor& b) : _w(w), _b(b), _w_grad(), _b_grad(), _ins() {
	_w_grad = zeros_like(_w);
	_b_grad = zeros_like(_b);
	_params = { &_w,&_b };
	_grads = { &_w_grad,&_b_grad };
}

std::vector<tensor> layer_linear::forward(const std::vector<tensor>& ins) {
	_ins = ins;
	std::vector<tensor> outs(ins.size());
	for (size_t i = 0; i < ins.size(); i++) {
		auto a = ins[i];
		auto h = matmul(a, _w);
		outs[i] = h;
		outs[i] += _b;
	}
	return outs;
}

std::vector<tensor> layer_linear::backward(const std::vector<tensor>& grads) {
	_w_grad.set_zero();
	_b_grad.set_zero();
	std::vector<tensor> outs(grads.size());
	for (size_t i = 0; i < grads.size(); i++) {
		_ins[i].reshape({_ins[i].size(), 1});
		_b_grad += grads[i];
		_w_grad += matmul(_ins[i], grads[i]);
		outs[i] = matmul_t(grads[i], _w);
	}
	return outs;
}
