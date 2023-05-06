#pragma once
#include "sample.h"
#include <vector>
class batch {
public:
	batch(std::vector<sample> samples);
	const std::vector<tensor>& xs() const;
	const std::vector<tensor>& ys() const;
	size_t size() const;

private:
	std::vector<tensor> _xs;
	std::vector<tensor> _ys;
};

