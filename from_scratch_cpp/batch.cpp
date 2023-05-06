#include "batch.h"

batch::batch(std::vector<sample> samples) : _xs(samples.size()), _ys(samples.size()) {
	for (size_t i = 0; i < samples.size(); i++) {
		_xs[i] = samples[i].x();
		_ys[i] = samples[i].y();
	}
}

const std::vector<tensor>& batch::xs() const {
	return _xs;
}

const std::vector<tensor>& batch::ys() const {
	return _ys;
}

size_t batch::size() const {
	return _xs.size();
}
