#include "loss_softmax_cross_entropy.h"

tensor softmax(const tensor& x) {
    tensor r(x);
    r.apply(exp);
    return r /= r.sum();
}

tensor cross_entropy(const tensor& probs, const tensor& y) {
    tensor r(probs);
    for (size_t i = 0; i < probs.size(); i++)
        r[i] = log(1e-30 + probs[i]) * y[i];
    return r;
}

std::tuple<double_t, tensor> loss_softmax_cross_entropy::loss_and_grad(const tensor& y_p, const tensor& y) {
    auto probs = softmax(y_p);
    return std::tuple<double_t, tensor>(cross_entropy(probs, y).sum(), probs - y);
}


