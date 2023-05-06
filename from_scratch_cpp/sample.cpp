#include "sample.h"

sample::sample() : _x(), _y() {}

sample::sample(const tensor& x, const tensor& y) : _x(x), _y(y) {}

const tensor& sample::x() const {
	return _x;
}

const tensor& sample::y() const {
	return _y;
}
