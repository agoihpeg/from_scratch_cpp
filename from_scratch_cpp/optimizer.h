#pragma once
#include "layer.h"
class optimizer {
public:
	optimizer(layer* model);
	virtual void step(size_t batch_size) = 0;
	virtual ~optimizer() {}

protected:
	size_t _params_count;
	layer* _model;
};
