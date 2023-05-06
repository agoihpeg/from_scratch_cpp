#pragma once
#include "loss.h"
class loss_softmax_cross_entropy : public loss {
public:
	std::tuple<double_t, tensor> loss_and_grad(const tensor& y_p, const tensor& y) override;
};
