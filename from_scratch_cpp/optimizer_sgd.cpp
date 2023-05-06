#include "optimizer_sgd.h"

optimizer_sgd::optimizer_sgd(layer* model, double_t lr) : optimizer(model), _lr(lr) {}

void optimizer_sgd::step(size_t batch_size) {
	auto p = _model->params().begin();
	auto g = _model->grads().begin();
	auto p_end = _model->params().end();

	for (; p != p_end; p++, g++)
		**p -= (**g) * _lr;
}
