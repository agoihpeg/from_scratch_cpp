#pragma once
#include "optimizer.h"
class optimizer_sgd : public optimizer {
public:
	optimizer_sgd(layer* model, double_t lr);
	void step(size_t batch_size) override;

private:
	double_t _lr;
};

