#include "optimizer.h"

optimizer::optimizer(layer* model) : _model(model), _params_count(0) {
	auto params = model->params();
	for (auto i = params.begin(); i != params.end(); i++)
		_params_count++;
}
