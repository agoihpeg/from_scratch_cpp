#pragma once
#include <vector>
#include <functional>

class tensor {
public:
	tensor();
	tensor(const std::vector<size_t>& shape, size_t size);
	tensor(const std::vector<size_t>& shape, const std::vector<double_t>& items);
	tensor get_range(const std::vector<size_t>& i0, const std::vector<size_t>& shape) const;
	void set_range(const std::vector<size_t>& i0, const std::vector<size_t>& shape, const tensor& t);
	void reshape(const std::vector<size_t>& shape);
	void fit_shape();
	void apply(std::function<double_t(double_t)> f);
	size_t size() const;
	bool empity() const;
	size_t dims() const;
	const std::vector<size_t>& shape() const;
	const std::vector<size_t>& offsets() const;
	double_t sum() const;
	void set(double_t val);
	void set_zero();
	const std::vector<double_t>& items() const;

	double_t& operator[](size_t i);
	const double_t& operator[](size_t i) const;
	double_t& operator[](const std::vector<size_t>& i);
	const double_t& operator[](const std::vector<size_t>& i) const;
	tensor& operator+=(const tensor& x);
	tensor& operator-=(const tensor& x);
	tensor& operator*=(const tensor& x);
	tensor& operator/=(const tensor& x);
	tensor& operator+=(double_t x);
	tensor& operator-=(double_t x);
	tensor& operator*=(double_t x);
	tensor& operator/=(double_t x);

private:
	std::vector<size_t> _shape;
	std::vector<size_t> _offsets;
	std::vector<double_t> _items;
};

tensor operator+(tensor x, const tensor& y);
tensor operator-(tensor x, const tensor& y);
tensor operator*(tensor x, const tensor& y);
tensor operator/(tensor x, const tensor& y);
tensor operator+(tensor x, double_t y);
tensor operator-(tensor x, double_t y);
tensor operator*(tensor x, double_t y);
tensor operator/(tensor x, double_t y);


tensor concat(const tensor& x, const tensor& y, size_t axis);
tensor repeat(std::vector<size_t> shape, double_t val);
tensor zeros(std::vector<size_t> shape);
tensor ones(std::vector<size_t> shape);
tensor zeros_like(const tensor& t);
//tensor tensordot(const tensor& x, const tensor& y, std::vector<size_t> axes1, std::vector<size_t> axes2);