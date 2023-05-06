#pragma once
#include "layer.h"
class layer_tanh : public layer {
public:
	std::vector<tensor> forward(const std::vector<tensor>& ins) override;
	std::vector<tensor> backward(const std::vector<tensor>& grads) override;

private:
	std::vector<tensor> _ins;
};

