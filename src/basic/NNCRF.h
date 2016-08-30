/*
 * NNCRF.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_NNCRF_H_
#define SRC_NNCRF_H_

#include <iostream>

#include "Example.h"
#include "Metric.h"
#include "N3L.h"

using namespace nr;
using namespace std;

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	vector<vector<LookupNode> > word_inputs;
	vector<ConcatNode> token_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;

	vector<BiNode> word_hidden2;
	vector<LinearNode> output;

	int type_num;

	//dropout nodes
	vector<vector<DropNode> > word_inputs_drop;
	vector<DropNode> word_hidden1_drop;
	vector<DropNode> word_hidden2_drop;

	// node pointers
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int typeNum){
		type_num = typeNum;
		resizeVec(word_inputs, sent_length, type_num + 1);		
		token_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);
		word_hidden2.resize(sent_length);
		output.resize(sent_length);

		resizeVec(word_inputs_drop, sent_length, type_num + 1);
		word_hidden1_drop.resize(sent_length);
		word_hidden2_drop.resize(sent_length);
	}

	inline void clear(){
		Graph::clear();
		clearVec(word_inputs);
		token_repsents.clear();
		word_window.clear();
		word_hidden1.clear();
		left_lstm.clear();
		right_lstm.clear();
		word_hidden2.clear();
		output.clear();	

		clearVec(word_inputs_drop);
		word_hidden1_drop.clear();
		word_hidden2_drop.clear();
	}

public:
	inline void initial(LookupTable& words, vector<LookupTable>& types, UniParams& tanh1_project, LSTM1Params& left_lstm_project,
		LSTM1Params& right_lstm_project, BiParams& tanh2_project, UniParams& olayer_linear, 
		int wordcontext, dtype dropout){
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&words);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&types[idy-1]);
			}

			for (int idy = 0; idy < word_inputs[idx].size(); idy++){
				word_inputs_drop[idx][idy].setDropValue(dropout);
			}

			word_hidden1[idx].setParam(&tanh1_project);
			word_hidden1[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden1_drop[idx].setDropValue(dropout);

			word_hidden2[idx].setParam(&tanh2_project);
			word_hidden2[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden2_drop[idx].setDropValue(dropout);
		}	
		word_window.setContext(wordcontext);
		left_lstm.setParam(&left_lstm_project, dropout, true);
		right_lstm.setParam(&right_lstm_project, dropout, false);

		for (int idx = 0; idx < output.size(); idx++){
			output[idx].setParam(&olayer_linear);
		}		
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(); // compute is a must step for train, predict and cost computation
		int seq_size = features.size();

		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx][0].forward(this, feature.words[0]);

			//drop out
			word_inputs_drop[idx][0].forward(this, &word_inputs[idx][0], bTrain);

			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].forward(this, feature.types[idy - 1]);
				//drop out
				word_inputs_drop[idx][idy].forward(this, &word_inputs[idx][idy], bTrain);
			}

			token_repsents[idx].forward(this, getPNodes(word_inputs_drop[idx], word_inputs_drop[idx].size()));
		}

		//windowlized
		word_window.forward(this, getPNodes(token_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(this, &(word_window._outputs[idx]));

			word_hidden1_drop[idx].forward(this, &word_hidden1[idx], bTrain);
		}

		left_lstm.forward(this, getPNodes(word_hidden1_drop, seq_size), bTrain);
		right_lstm.forward(this, getPNodes(word_hidden1_drop, seq_size), bTrain);

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden2[idx].forward(this, &(left_lstm._hiddens_drop[idx]), &(right_lstm._hiddens_drop[idx]));

			word_hidden2_drop[idx].forward(this, &word_hidden2[idx], bTrain);

			output[idx].forward(this, &(word_hidden2_drop[idx]));
		}
	}

};

//A native neural network classfier using only word embeddings

class NNCRF {
public:
	NNCRF() {
		_dropOut = 0.0;
		_pcg = NULL;
		_types.clear();
	}

	~NNCRF() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	LookupTable _words;
	vector<LookupTable> _types;

	int _wordcontext, _wordwindow;
	int _wordDim;
	vector<int> _typeDims;
	int _unitsize;


	int _hiddensize;
	int _rnnhiddensize;
	int _inputsize;
	LSTM1Params _left_lstm_project; //left lstm
	LSTM1Params _right_lstm_project; //right lstm
	UniParams _tanh1_project; // hidden
	BiParams _tanh2_project; // hidden
	UniParams _olayer_linear; // output
	

	CRFMLLoss _loss;

	int _labelSize;

	Metric _eval;

	dtype _dropOut;

	ModelUpdate _ada;

	ComputionGraph *_pcg;

	CheckGrad _checkgrad;

public:
	//embeddings are initialized before this separately.
	inline void init(int wordcontext, int charcontext, int hiddensize, int rnnhiddensize, int labelSize) {
		if (_words.nVSize <= 0){
			std::cout << "Please initialize embeddings before this" << std::endl;
			return;
		}
		_wordcontext = wordcontext;
		_wordwindow = 2 * _wordcontext + 1;
		_wordDim = _words.nDim;
		_unitsize = _wordDim;
		_typeDims.clear();
		for (int idx = 0; idx < _types.size(); idx++){
			_typeDims.push_back(_types[idx].nDim);
			_unitsize += _typeDims[idx];
		}


		_labelSize = labelSize;
		_hiddensize = hiddensize;
		_rnnhiddensize = rnnhiddensize;
		_inputsize = _wordwindow * _unitsize;


		_tanh1_project.initial(_hiddensize, _inputsize, true, 100);
		_left_lstm_project.initial(rnnhiddensize, _hiddensize, 200);
		_right_lstm_project.initial(rnnhiddensize, _hiddensize, 300);
		_tanh2_project.initial(_hiddensize, rnnhiddensize, rnnhiddensize, true, 400);
		_olayer_linear.initial(_labelSize, _hiddensize, false, 600);

		assert(_loss.labelSize == _labelSize);
		_loss.initial(_labelSize, 700);

		//ada
		_words.exportAdaParams(_ada);
		for (int idx = 0; idx < _types.size(); idx++){
			_types[idx].exportAdaParams(_ada);
		}
		_tanh1_project.exportAdaParams(_ada);
		_left_lstm_project.exportAdaParams(_ada);
		_right_lstm_project.exportAdaParams(_ada);
		_tanh2_project.exportAdaParams(_ada);
		_olayer_linear.exportAdaParams(_ada);
		_loss.exportAdaParams(_ada);


		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length, _types.size());
		_pcg->initial(_words, _types, _tanh1_project, _left_lstm_project, _right_lstm_project, _tanh2_project, _olayer_linear, _wordcontext, _dropOut);

		//check grad
		_checkgrad.add(&(_words.E), "_words.E");
		for (int idx = 0; idx < _types.size(); idx++){
			stringstream ss;
			ss << "_types[" << idx << "].E";
			_checkgrad.add(&(_types[idx].E), ss.str());
		}
		_checkgrad.add(&(_tanh1_project.W), "_tanh1_project.W");
		_checkgrad.add(&(_tanh1_project.b), "_tanh1_project.b");


		_checkgrad.add(&(_tanh2_project.W1), "_tanh2_project.W1");
		_checkgrad.add(&(_tanh2_project.W2), "_tanh2_project.W2");
		_checkgrad.add(&(_tanh2_project.b), "_tanh2_project.b");



		_checkgrad.add(&(_olayer_linear.W), "_olayer_linear.W");
		//_checkgrad.add(&(_loss.T), "_loss.T");

		//if (_ada._params.size() != _checkgrad._params.size()){
		//	std::cout << "_ada._params: " << _ada._params.size() << ",  _checkgrad._params: " << _checkgrad._params.size() << std::endl;
		//}
	}


	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		static vector<PMat> tpmats;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example.m_features, true); 

			//loss function
			int seq_size = example.m_features.size();
			//for (int idx = 0; idx < seq_size; idx++) {
				//cost += _loss.loss(&(_pcg->output[idx]), example.m_labels[idx], _eval, example_num);				
			//}
			cost += _loss.loss(getPNodes(_pcg->output, seq_size), example.m_labels, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature>& features, vector<int>& results) {
		_pcg->forward(features);
		int seq_size = features.size();
		//results.resize(seq_size);
		//for (int idx = 0; idx < seq_size; idx++) {
		//	_loss.predict( &(_pcg->output[idx]), results[idx]);
		//}
		_loss.predict(getPNodes(_pcg->output, seq_size), results);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_features, true); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		//loss function
		//for (int idx = 0; idx < seq_size; idx++) {
		//	cost += _loss.cost(&(_pcg->output[idx]), example.m_labels[idx], 1);
		//}
		cost += _loss.cost(getPNodes(_pcg->output, seq_size), example.m_labels, 1);

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



public:
	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_NNCRF_H_ */
