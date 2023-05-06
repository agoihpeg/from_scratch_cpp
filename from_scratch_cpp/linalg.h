#pragma once
#include "tensor.h"

tensor matmul(const tensor& x, const tensor& y);
tensor matmul_t(const tensor& x, const tensor& y);
double_t dot(const tensor& x, const tensor& y);
