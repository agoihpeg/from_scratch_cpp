#pragma once
#include "tensor.h"
class sample {
public:
	sample();
	sample(const tensor& x, const tensor& y);
	const tensor& x() const;
	const tensor& y() const;

private:
	tensor _x;
	tensor _y;
};