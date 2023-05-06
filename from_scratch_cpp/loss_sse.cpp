#include "loss_sse.h"

std::tuple<double_t, tensor> loss_sse::loss_and_grad(const tensor& y_p, const tensor& y) {
	auto errs = y_p - y;
	auto grad = errs * 2;
	errs *= errs;
	double_t loss = errs.sum();
	return std::tuple<double_t, tensor>(loss, grad);
}
