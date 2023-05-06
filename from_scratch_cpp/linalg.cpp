#include "linalg.h"
#include <assert.h>

tensor matmul(const tensor& x, const tensor& y) {
	size_t h1 = x.shape()[0], w1 = x.shape()[1], h2 = y.shape()[0], w2 = y.shape()[1],
		size1 = x.size(), size2 = y.size(), rsize = h1 * w2;
	assert(x.dims() == 2);
	assert(y.dims() == 2);
	assert(w1 == h2);
	tensor z({ h1, w2 }, h1 * w2);
	size_t i = 0;
	for (size_t row_offset = 0; row_offset < size1; row_offset += w1)
		for (size_t col_offset = 0; col_offset < w2; col_offset++, i++)
			for (size_t row_iter = row_offset, col_iter = col_offset; col_iter < size2; row_iter++, col_iter += w2)
				z[i] += x[row_iter] * y[col_iter];
	return z;
}

tensor matmul_t(const tensor& x, const tensor& y) {
	size_t h1 = x.shape()[0], w1 = x.shape()[1], h2 = y.shape()[0], w2 = y.shape()[1],
		size1 = x.size(), size2 = y.size(), rsize = h1 * w2;
	assert(x.dims() == 2);
	assert(y.dims() == 2);
	assert(w1 == w2);
	tensor z({ h1, h2 }, h1 * h2);
	size_t i = 0;
	for (size_t row1_offset = 0; row1_offset < size1; row1_offset += w1)
		for (size_t row2_offset = 0; row2_offset < size2; row2_offset += w2, i++)
			for (size_t row1_iter = row1_offset, row2_iter = row2_offset, j = 0; j < w1; row1_iter++, row2_iter++, j++)
				z[i] += x[row1_iter] * y[row2_iter];
	return z;
}

double_t dot(const tensor& x, const tensor& y) {
	assert(x.shape() == y.shape());
	double_t z = 0;
	for (size_t i = 0; i < x.size(); i++)
		z += x[i] * y[i];
	return z;
}
