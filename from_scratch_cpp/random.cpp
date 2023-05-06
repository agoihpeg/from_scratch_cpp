#include "random.h"
#include "probs.h"
#include "sample.h"
#include <random>

std::default_random_engine rng;

void seed(unsigned int seed) {
	rng = std::default_random_engine(seed);
}

void shuffle(std::vector<sample>& v) {
	std::shuffle(v.begin(), v.end(), rng);
}

std::tuple<std::vector<sample>, std::vector<sample>> train_test_split(std::vector<sample> v, double_t train_part) {
	shuffle(v);
	size_t i = (size_t)(v.size() * train_part);
	return std::tuple<std::vector<sample>, std::vector<sample>>(
		std::vector<sample>(v.begin(), v.begin() + i),
		std::vector<sample>(v.begin() + i, v.end()));
}

size_t sum(const std::vector<size_t>& v) {
	size_t r = 0;
	for (auto& i : v) r += i;
	return r;
}

double_t random_uniform() {
	return (double_t)rng() / std::default_random_engine::_Max;
}

double_t random_normal(double_t mean, double_t variance) {
	return inverse_normal_cdf(random_uniform(), mean, variance);
}

tensor random_tensor(const std::vector<size_t>& shape, distribution d) {
	size_t size = 1;
	for (auto& s : shape) size *= s;
	std::vector<double_t> items(size);
	switch (d)
	{
	case distribution::uniform:
		for (auto& i : items) i = random_uniform();
		break;
	case distribution::normal:
		for (auto& i : items) i = random_normal();
		break;
	case distribution::xavier:
		double_t variance = (double_t)shape.size() / sum(shape);
		for (auto& i : items) i = random_normal(variance = variance);
		break;
	}
	return tensor(shape, items);
}
