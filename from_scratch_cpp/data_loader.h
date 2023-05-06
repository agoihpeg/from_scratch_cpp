#pragma once
#include "batch.h"
class data_loader {
public:
	class batch_iterator {
	public:
		batch_iterator(std::vector<sample>::const_iterator it, size_t batch_size, size_t data_size, size_t index);
		batch_iterator& operator++();
		bool operator!=(const batch_iterator& b);
		batch operator*();

	private:
		std::vector<sample>::const_iterator _it;
		size_t _batch_size;
		size_t _data_size;
		size_t _index;
	};
	data_loader(std::vector<sample> training_data, size_t batch_size);
	batch_iterator begin();
	batch_iterator end();

private:
	std::vector<sample> _training_data;
	size_t _batch_size;
};
