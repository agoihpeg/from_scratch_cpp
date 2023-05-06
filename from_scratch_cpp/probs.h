#pragma once
#include <cmath>

double_t normal_cdf(double_t x, double_t mu = 0, double_t sigma = 1.0);
double_t inverse_normal_cdf(double_t p, double_t mu = 0, double_t sigma = 1.0, double_t tol = 1e-5);
