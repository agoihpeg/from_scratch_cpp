#include "sequential.h"

sequential::sequential(std::list<layer*> layers) : _layers(layers) {
	std::forward_list<tensor*> prm, grd;
	for (auto it = _layers.rbegin(); it != _layers.rend(); it++) {
		prm.splice_after(prm.before_begin(), (*it)->params());
		grd.splice_after(grd.before_begin(), (*it)->grads());
	}
	_params = prm;
	_grads = grd;
}

sequential::~sequential() {
	for (auto l : _layers) delete l;
}

std::vector<tensor> sequential::forward(const std::vector<tensor>& ins) {
	std::vector<tensor> buff = ins;
	for (auto it = _layers.begin(); it != _layers.end(); it++)
		buff = (*it)->forward(buff);
	return buff;
}

std::vector<tensor> sequential::backward(const std::vector<tensor>& grads) {
	std::vector<tensor> buff = grads;
	for (auto it = _layers.rbegin(); it != _layers.rend(); it++)
		buff = (*it)->backward(buff);
	return buff;
}
