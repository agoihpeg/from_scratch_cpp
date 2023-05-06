#pragma once
#include <vector>
#include <forward_list>
#include "tensor.h"
#include "batch.h"
class layer {
public:
	virtual std::vector<tensor> forward(const std::vector<tensor>& ins) = 0;
	virtual std::vector<tensor> backward(const std::vector<tensor>& grads) = 0;
	virtual ~layer() {};
	std::forward_list<tensor*>& params();
	std::forward_list<tensor*>& grads();

protected:
	std::forward_list<tensor*> _params;
	std::forward_list<tensor*> _grads;
};
