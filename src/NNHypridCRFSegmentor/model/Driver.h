#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"

class Driver{
public:
	Driver(size_t memsize): _aligned_mem(memsize){

	}
	~Driver(){

	}
public:
	ComputionGraph* _pcg;
	ModelParams _model_params;
	HyperParams _hyper_params;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;
	AlignedMemoryPool _aligned_mem;

public:
	inline void initial(){
		if (!_hyper_params.bVaild()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}

		if (!_model_params.initial(_hyper_params)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_model_params.exportModelParams(_ada);
		_model_params.exportCheckGradParams(_checkgrad);
		_hyper_params.print();

		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length, _hyper_params.maxsegLen, _model_params._types.size());
		_pcg->initial(_model_params, _hyper_params, &_aligned_mem);

		setUpdateParameters(_hyper_params.nnRegular, _hyper_params.adaAlpha, _hyper_params.adaEps);

	}

	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example.m_features, true);

			//loss function
			int seq_size = example.m_features.size();
			//for (int idx = 0; idx < seq_size; idx++) {
			//cost += _loss.loss(&(_pcg->output[idx]), example.m_labels[idx], _eval, example_num);				
			//}
			cost += _model_params._loss.loss(_pcg->poutput, example.m_seglabels, _eval, example_num);
			cost += _model_params._bmesloss.loss(getPNodes(_pcg->output_bmes, seq_size), example.m_labels, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature>& features, NRMat<int>& results, vector<int>& result2s) {
		_pcg->forward(features);
		int seq_size = features.size();
		//results.resize(seq_size);
		//for (int idx = 0; idx < seq_size; idx++) {
		//	_loss.predict( &(_pcg->output[idx]), results[idx]);
		//}
		_model_params._loss.predict(_pcg->poutput, results);
		_model_params._bmesloss.predict(getPNodes(_pcg->output_bmes, seq_size), result2s);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_features, true); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		//loss function
		//for (int idx = 0; idx < seq_size; idx++) {
		//	cost += _loss.cost(&(_pcg->output[idx]), example.m_labels[idx], 1);
		//}
		cost += _model_params._loss.cost(_pcg->poutput, example.m_seglabels, 1);
		cost += _model_params._bmesloss.cost(getPNodes(_pcg->output_bmes, seq_size), example.m_labels, 1);

		return cost;
	}


	void updateModel() {
		//_ada.update();
		_ada.update(5.0);
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void writeModel();

	void loadModel();
};
#endif /*SRC_Driver_H_*/