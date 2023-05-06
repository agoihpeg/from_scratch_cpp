#include "loss.h"

std::tuple<double_t, std::vector<tensor>> loss::loss_and_grad(const std::vector<tensor>& ys_p, const std::vector<tensor>& ys) {
	std::vector<tensor> grads(ys.size());
	double_t ls = 0, bf = 0;
	for (size_t i = 0; i < ys.size(); i++) {
		std::tie(bf, grads[i]) = loss_and_grad(ys_p[i], ys[i]);
		ls += bf;
	}
	return std::tuple<double_t, std::vector<tensor>>(ls, grads);
}
