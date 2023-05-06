#pragma once
#include "tensor.h"
class loss {
public:
	virtual std::tuple<double_t, tensor> loss_and_grad(const tensor& y_p, const tensor& y) = 0;
	std::tuple<double_t, std::vector<tensor>> loss_and_grad(const std::vector<tensor>& ys_p, const std::vector<tensor>& ys);
	virtual ~loss() {}
};

