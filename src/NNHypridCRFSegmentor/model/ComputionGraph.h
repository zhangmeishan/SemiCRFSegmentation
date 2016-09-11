#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "BMESSegmentation.h"

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
	vector<SegBuilder> outputseg;
	vector<LinearNode> output;
	vector<LinearNode> output_bmes;

	NRMat<PNode> poutput; //use to store pointer matrix of outputs
	int max_seg_length;
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
	inline void createNodes(int sent_length, int maxsegLen, int typeNum){
		max_seg_length = maxsegLen;
		type_num = typeNum;
		int segNum = sent_length * max_seg_length;
		resizeVec(word_inputs, sent_length, type_num + 1);
		token_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);
		word_hidden2.resize(sent_length);
		output_bmes.resize(sent_length);
		outputseg.resize(segNum);
		for (int idx = 0; idx < segNum; idx++){
			outputseg[idx].resize(maxsegLen);
		}
		output.resize(segNum);

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
		outputseg.clear();
		output.clear();
		output_bmes.clear();

		clearVec(word_inputs_drop);
		word_hidden1_drop.clear();
		word_hidden2_drop.clear();
	}

public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params){
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&model_params._words);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&model_params._types[idy - 1]);
			}

			for (int idy = 0; idy < word_inputs[idx].size(); idy++){
				word_inputs_drop[idx][idy].setDropValue(hyper_params.dropProb);
			}

			word_hidden1[idx].setParam(&model_params._tanh1_project);
			word_hidden1[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden1_drop[idx].setDropValue(hyper_params.dropProb);

			word_hidden2[idx].setParam(&model_params._tanh2_project);
			word_hidden2[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden2_drop[idx].setDropValue(hyper_params.dropProb);
			output_bmes[idx].setParam(&model_params._olayerbmes_linear);
		}
		word_window.setContext(hyper_params.wordContext);
		left_lstm.setParam(&model_params._left_lstm_project, hyper_params.dropProb, true);
		right_lstm.setParam(&model_params._right_lstm_project, hyper_params.dropProb, false);

		for (int idx = 0; idx < output.size(); idx++){
			outputseg[idx].setParam(&model_params._seglayer_project, hyper_params.dropProb);
			outputseg[idx].setFunctions(&tanh, &tanh_deri);
			output[idx].setParam(&model_params._olayer_linear);
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

			output_bmes[idx].forward(this, &word_hidden2_drop[idx]);
		}

		static int offset;
		vector<PNode> segnodes;
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			segnodes.clear();
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				segnodes.push_back(&word_hidden2_drop[idx + dist]);
				outputseg[offset + dist].forward(this, segnodes, bTrain);
			}
		}

		poutput.resize(seq_size, max_seg_length);
		poutput = NULL;
		offset = 0;
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				output[offset + dist].forward(this, &(outputseg[offset + dist]._output_drop));
				poutput[idx][dist] = &output[offset + dist];
			}
		}

		for (int idx = 0; idx < seq_size; idx++) {
			for (int dist = 0; dist < max_seg_length; dist++) {
				if (poutput[idx][dist] != NULL){
					exportNode(poutput[idx][dist]);
				}
			}
			exportNode(&output_bmes[idx]);
		}
	}

};

#endif /*SRC_ComputionGraph_H_*/