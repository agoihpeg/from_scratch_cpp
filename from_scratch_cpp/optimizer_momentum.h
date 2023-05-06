#pragma once
#include "optimizer.h"
class optimizer_momentum : public optimizer {
public:
	optimizer_momentum(layer* model);
	void step(size_t batch_size) override;

private:
	std::forward_list<tensor> _moments;
};
