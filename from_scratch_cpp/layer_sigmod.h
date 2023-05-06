#pragma once
#include <vector>
#include "tensor.h"
#include "layer.h"

class layer_sigmod : public layer {
public:
	std::vector<tensor> forward(const std::vector<tensor>& ins) override;
	std::vector<tensor> backward(const std::vector<tensor>& grads) override;

private:
	std::vector<tensor> _ins;
};

