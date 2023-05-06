#include "layer.h"

std::forward_list<tensor*>& layer::params() {
    return _params;
}

std::forward_list<tensor*>& layer::grads() {
    return _grads;
}
