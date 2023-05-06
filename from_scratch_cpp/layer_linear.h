#pragma once
#include "layer.h"
#include "linalg.h"
#include "random.h"

class layer_linear : public layer {
public:
	layer_linear(size_t in, size_t out, distribution d);
	layer_linear(const tensor& w, const tensor& b);
	std::vector<tensor> forward(const std::vector<tensor>& ins) override;
	std::vector<tensor> backward(const std::vector<tensor>& grads) override;

private:
	tensor _w, _b;
	tensor _w_grad, _b_grad;
	std::vector<tensor> _ins;
};
