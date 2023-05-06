#pragma once
#include <list>
#include "layer.h"
class sequential : public layer {
public:
	sequential(std::list<layer*> layers);
	~sequential();
	std::vector<tensor> forward(const std::vector<tensor>& ins) override;
	std::vector<tensor> backward(const std::vector<tensor>& grads) override;

private:
	std::list<layer*> _layers;
};

