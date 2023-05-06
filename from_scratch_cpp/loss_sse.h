#pragma once
#include "loss.h"
class loss_sse : public loss {
public:
	std::tuple<double_t, tensor> loss_and_grad(const tensor& y_p, const tensor& y);
};

