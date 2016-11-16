#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
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
	vector<UniNode> word_hidden3;
	vector<LinearNode> output;

	int type_num;


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
		word_hidden3.resize(sent_length);
		output.resize(sent_length);

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
		word_hidden3.clear();
		output.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem){
		int maxsize = word_inputs.size();
		for (int idx = 0; idx < maxsize; idx++) {
			word_inputs[idx][0].setParam(&model.words);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&model.types[idy - 1]);
			}
			word_hidden1[idx].setParam(&model.tanh1_project);
			word_hidden2[idx].setParam(&model.tanh2_project);
			word_hidden3[idx].setParam(&model.tanh3_project);
			output[idx].setParam(&model.olayer_linear);
		}
		
		word_window.init(opts.unitsize, opts.wordcontext, mem);
		left_lstm.init(&model.left_lstm_project, opts.dropOut, true, mem);
		right_lstm.init(&model.right_lstm_project, opts.dropOut, false, mem);

		for (int idx = 0; idx < maxsize; idx++){
			word_inputs[idx][0].init(opts.wordDim, opts.dropOut, mem);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].init(opts.typeDims[idy-1], opts.dropOut, mem);
			}
			token_repsents[idx].init(opts.unitsize, -1, mem);
			word_hidden1[idx].init(opts.hiddensize, opts.dropOut, mem);
			word_hidden2[idx].init(opts.hiddensize, opts.dropOut, mem);
			word_hidden3[idx].init(opts.hiddensize, -1, mem);
			output[idx].init(opts.labelSize, -1, mem);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		// second step: build graph
		int seq_size = features.size();
		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx][0].forward(this, feature.words[0]);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].forward(this, feature.types[idy - 1]);
			}

			token_repsents[idx].forward(this, getPNodes(word_inputs[idx], word_inputs[idx].size()));
		}

		//windowlized
		word_window.forward(this, getPNodes(token_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(this, &(word_window._outputs[idx]));
		}

		left_lstm.forward(this, getPNodes(word_hidden1, seq_size));
		right_lstm.forward(this, getPNodes(word_hidden1, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden2[idx].forward(this, &(left_lstm._hiddens[idx]), &(right_lstm._hiddens[idx]));
			word_hidden3[idx].forward(this, &(word_hidden2[idx]));
			output[idx].forward(this, &(word_hidden3[idx]));
		}
	}

};

#endif /* SRC_ComputionGraph_H_ */