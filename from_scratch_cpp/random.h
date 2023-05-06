#pragma once
#include "tensor.h"
#include "sample.h"
enum class distribution {
	uniform,
	normal,
	xavier
};

void seed(unsigned int seed);
void shuffle(std::vector<sample>& v);
std::tuple<std::vector<sample>, std::vector<sample>> train_test_split(std::vector<sample> v, double_t train_part);
double_t random_uniform();
double_t random_normal(double_t mean = 0, double_t variance = 1);
tensor random_tensor(const std::vector<size_t>& shape, distribution d);