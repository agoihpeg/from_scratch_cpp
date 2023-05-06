#include <iostream>
#include "layer_linear.h"
#include "sequential.h"
#include "layer_sigmod.h"
#include "optimizer_sgd.h"
#include "loss_sse.h"
#include "data_loader.h"

tensor vector(std::initializer_list<double_t> l) {
	return tensor({ 1, l.size() }, std::vector<double_t>(l));
}


int main() {
	std::vector<sample> data{
		sample(vector({0, 0}), vector({0})),
		sample(vector({1, 0}), vector({1})),
		sample(vector({0, 1}), vector({1})),
		sample(vector({1, 1}), vector({0}))
	};
	seed(42);
	data_loader loader(data, 1);
	auto model = new sequential({
		new layer_linear(2, 2, distribution::xavier),
		new layer_sigmod(),
		new layer_linear(2, 1, distribution::xavier) });
	optimizer_sgd optimizer(model, 0.1);
	loss* loss = new loss_sse();
	size_t epoches = 3000;
	std::vector<double_t> losses(epoches);
	std::vector<tensor> ys_p;
	std::vector<tensor> grads;
	double_t batch_loss, epoch_loss;

	for (size_t i = 0; i < epoches; i++) {
		epoch_loss = 0;
		for (auto batches = loader.begin(); batches != loader.end(); ++batches) {
			batch_loss = 0;
			batch batch = *batches;
			ys_p  = model->forward(batch.xs());
			std::tie(batch_loss, grads) = loss->loss_and_grad(ys_p, batch.ys());
			model->backward(grads);
			optimizer.step(batch.size());
			epoch_loss += batch_loss;
		}
		losses[i] = epoch_loss;
	}
	for (size_t i = 0; i < losses.size(); i += 100)
		std::cout << i + 1 << ". " << losses[i] << std::endl;
	for (size_t i = 0; i < 4; i++) {
		std::cout << data[i].x()[0] << " xor " << data[i].x()[1] << " = ";
		std::cout << model->forward({ data[i].x() })[0][0] << std::endl;
	}
	delete loss;
	delete model;
}
