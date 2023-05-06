#include "data_loader.h"
#include "random.h"

data_loader::batch_iterator::batch_iterator(std::vector<sample>::const_iterator it, size_t batch_size, size_t data_size, size_t index)
	: _it(it), _batch_size(batch_size), _data_size(data_size), _index(index) {}

data_loader::batch_iterator& data_loader::batch_iterator::operator++(){
	size_t new_index = std::min(_index + _batch_size, _data_size);
	size_t size = new_index - _index;
	this->_index = new_index;
	_it += size;
	return *this;
}

bool data_loader::batch_iterator::operator!=(const batch_iterator& b) {
	 return this->_it != b._it;
}

batch data_loader::batch_iterator::operator*() {
	size_t new_index = std::min(_index + _batch_size, _data_size - 1);
	size_t size = new_index - _index;
	std::vector<sample> samples(size);
	std::copy(_it, _it + size, samples.begin());
	return batch(samples);
}


data_loader::data_loader(std::vector<sample> training_data, size_t batch_size)
	: _training_data(training_data), _batch_size(batch_size) {}

data_loader::batch_iterator data_loader::begin() {
	shuffle(_training_data);
	return batch_iterator(_training_data.begin(), _batch_size, _training_data.size(), 0);
}

data_loader::batch_iterator data_loader::end() {
	return batch_iterator(_training_data.end(), _batch_size, _training_data.size(), _training_data.size() / _batch_size);
}


