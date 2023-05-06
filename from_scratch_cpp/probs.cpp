#include "probs.h"

double_t normal_cdf(double_t x, double_t mu, double_t sigma) {
	double_t n = (x - mu) / sigma / sqrt(2);
	return (1 + erf(n)) / 2;
}

double_t inverse_normal_cdf(double_t p, double_t mu, double_t sigma, double_t tol) {
	if (mu != 0 || sigma != 1)
		return mu + sigma * inverse_normal_cdf(p, 0, 1, tol);
	double_t lo = -10, hi = 10, mid;
	do {
		mid = (hi + lo) / 2;
		double_t midp = normal_cdf(mid);
		if (midp < p) lo = mid;
		else hi = mid;
	} while (hi - lo > tol);
	return mid;
}
