#include "optimizer_momentum.h"

optimizer_momentum::optimizer_momentum(layer* model) : optimizer(model) {
	_moments = std::forward_list<tensor>(_params_count);
	auto prm_it = _model->params().begin();
	for (auto mom_it = _moments.begin(); mom_it  != _moments.end(); mom_it++, prm_it++)
	{
		*mom_it = zeros_like(**prm_it);
	}
}

void optimizer_momentum::step(size_t batch_size) {

}
