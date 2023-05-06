#include <assert.h>
#include "tensor.h"
#include <ostream>
#include <algorithm>

std::vector<size_t> calc_offsets(const std::vector<size_t>& shape) {
	if (shape.empty()) return std::vector<size_t>();
	std::vector<size_t> offsets(shape);
	size_t x = offsets.back();
	offsets.back() = 1;
	for (auto i = offsets.rbegin() + 1; i != offsets.rend(); i++) {
		size_t t = *i;
		*i = x;
		x *= t;
	}
	return offsets;
}

size_t mult(const std::vector<size_t> x) {
	if (x.empty()) return 0;
	size_t z = 1;
	for (auto& i : x) z *= i;
	return z;
}

size_t dot(const std::vector<size_t>& x, const std::vector<size_t>& y) {
	assert(x.size() == y.size());
	size_t r = 0;
	for (size_t i = 0; i < y.size(); i++)
		r += x[i] * y[i];
	return r;
}

std::vector<size_t> sub(const std::vector<size_t>& x, const std::vector<size_t>& y) {
	assert(x.size() == y.size());
	std::vector<size_t> r(x);
	for (size_t i = 0; i < x.size(); i++)
		r[i] -= y[i];
	return r;
}
void add(std::vector<size_t>& x, size_t y) {
	for (size_t i = 0; i < x.size(); i++)
		x[i] += y;
}

void _copy_range(tensor& to, const tensor& from, const std::vector<size_t>& ito, const std::vector<size_t>& ifrom,
	const std::vector<size_t>& shape, size_t t, size_t f, size_t dim) {
	if (to.dims() - 1 == dim)
		for (size_t j = 0; j < shape[dim]; j++, t++, f++)
			to[t] = from[f];
	else
		for (size_t j = 0; j < shape[dim]; j++, t+=to.offsets()[dim], f+=from.offsets()[dim])
			_copy_range(to, from, ito, ifrom, shape, t, f, dim + 1);
}

void copy_range(tensor& to, const tensor& from, const std::vector<size_t>& ito, const std::vector<size_t>& ifrom, const std::vector<size_t>& shape) {
	size_t t = dot(to.offsets(), ito), f = dot(from.offsets(), ifrom);
	_copy_range(to, from, ito, ifrom, shape, t, f, 0);
}



tensor::tensor() : _shape(), _offsets(), _items() {}

tensor::tensor(const std::vector<size_t>& shape, size_t size) : _shape(shape), _offsets(calc_offsets(shape)), _items(size) {}

tensor::tensor(const std::vector<size_t>& shape, const std::vector<double_t>& items) : 
	_shape(shape), _offsets(calc_offsets(shape)), _items(items) {}

tensor tensor::get_range(const std::vector<size_t>& i0, const std::vector<size_t>& shape) const {
	size_t rsize = mult(shape);
	std::vector<double_t> ritems(rsize);
	tensor r(shape, ritems);
	std::vector<size_t> ito(shape.size());
	copy_range(r, *this, ito, i0, shape);
	return r;
}

void tensor::set_range(const std::vector<size_t>& i0, const std::vector<size_t>& shape, const tensor& t) {
	std::vector<size_t> ifrom(t._shape.size());
	copy_range(*this, t, i0, ifrom, shape);
}

void tensor::reshape(const std::vector<size_t>& shape) {
	_shape = shape;
	_offsets = calc_offsets(shape);
}

void tensor::fit_shape() {
	size_t j = 0;
	for (size_t i = 0; i < _shape.size(); i++)
		if (_shape[j] != 1)
			_shape[j++] = _shape[i];
}

void tensor::apply(std::function<double_t(double_t)> f) {
	for (auto& i : _items) i = f(i);
}

size_t tensor::size() const {
	return _items.size();
}

bool tensor::empity() const {
	return _items.empty();
}

size_t tensor::dims() const {
	return _shape.size();
}

const std::vector<size_t>& tensor::shape() const {
	return _shape;
}

const std::vector<size_t>& tensor::offsets() const {
	return _offsets;
}

double_t tensor::sum() const {
	double_t r = 0;
	for (auto& i : _items) r += i;
	return r;
}

void tensor::set(double_t val) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] = val;
}

void tensor::set_zero() {
	set(0);
}

const std::vector<double_t>& tensor::items() const {
	return _items;
}


double_t& tensor::operator[](size_t i) {
	return _items[i];
}

const double_t& tensor::operator[](size_t i) const {
	return _items[i];
}

double_t& tensor::operator[](const std::vector<size_t>& i) {
	assert(i.size() == _offsets.size());
	return _items[dot(i, _offsets)];
}

const double_t& tensor::operator[](const std::vector<size_t>& i) const {
	assert(i.size() == _offsets.size());
	return _items[dot(i, _offsets)];
}

tensor& tensor::operator+=(const tensor& x) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] += x._items[i];
	return *this;
}

tensor& tensor::operator-=(const tensor& x) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] -= x._items[i];
	return *this;
}

tensor& tensor::operator*=(const tensor& x) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] *= x._items[i];
	return *this;
}

tensor& tensor::operator/=(const tensor& x) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] /= x._items[i];
	return *this;
}

tensor& tensor::operator+=(double_t x) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] += x;
	return *this;
}

tensor& tensor::operator-=(double_t x) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] -= x;
	return *this;
}

tensor& tensor::operator*=(double_t x) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] *= x;
	return *this;
}

tensor& tensor::operator/=(double_t x) {
	for (size_t i = 0; i < _items.size(); i++)
		_items[i] /= x;
	return *this;
}

tensor operator+(tensor x, const tensor& y) {
	return x += y;
}

tensor operator-(tensor x, const tensor& y) {
	return x -= y;
}

tensor operator*(tensor x, const tensor& y) {
	return x *= y;
}

tensor operator/(tensor x, const tensor& y) {
	return x /= y;
}

tensor operator+(tensor x, double_t y) {
	return x += y;
}

tensor operator-(tensor x, double_t y) {
	return x -= y;
}

tensor operator*(tensor x, double_t y) {
	return x *= y;
}

tensor operator/(tensor x, double_t y) {
	return x /= y;
}


tensor concat(const tensor& x, const tensor& y, size_t axis) {
	assert(x.dims() == y.dims());
	for (size_t i = 0; i < x.dims(); i++)
		if (i != axis)
			assert(x.shape()[i] == y.shape()[i]);
	std::vector<size_t> rshape = x.shape();
	rshape[axis] += y.shape()[axis];
	size_t rsize = x.size() + y.size();
	tensor r(rshape, rsize);
	std::vector<size_t> i0(rshape.size());
	r.set_range(i0, x.shape(), x);
	i0[axis] += x.shape()[axis];
	r.set_range(i0, y.shape(), y);
	return r;
}

tensor repeat(std::vector<size_t> shape, double_t val) {
	size_t size = mult(shape);
	std::vector<double_t> items(size, val);
	return tensor(shape, items);
}

tensor zeros(std::vector<size_t> shape) {
	return repeat(shape, 0);
}

tensor ones(std::vector<size_t> shape) {
	return repeat(shape, 1);
}

tensor zeros_like(const tensor& t) {
	return zeros(t.shape());
}

//tensor tensordot(const tensor& x, const tensor& y, std::vector<size_t> axes1, std::vector<size_t> axes2) {
//	assert(axes1.size() == axes2.size());
//	for (size_t i = 0; i < axes1.size(); i++)
//		assert(x.shape()[i] == y.shape()[i]);
//	size_t k = 0;
//	throw "not implemented";
//}
