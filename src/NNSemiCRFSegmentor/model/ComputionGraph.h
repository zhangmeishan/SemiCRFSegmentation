#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Segmentation.h"

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	int max_seg_length;
	int type_num;

	vector<vector<LookupNode> > word_inputs;
	vector<ConcatNode> token_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;

	vector<BiNode> word_hidden2;
	vector<SegBuilder> outputseg;
	vector<LinearNode> output;

	NRMat<PNode> poutput; //use to store pointer matrix of outputs

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
		outputseg.resize(segNum);
		for (int idx = 0; idx < segNum; idx++){
			outputseg[idx].resize(maxsegLen);
		}
		output.resize(segNum);
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
	}


public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&model_params._words);
			word_inputs[idx][0].init(hyper_params.wordDim, hyper_params.dropProb, mem);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&model_params._types[idy-1]);
				word_inputs[idx][idy].init(hyper_params.wordDim, hyper_params.dropProb, mem);
			}

			token_repsents[idx].init(hyper_params.unitSize, -1, mem);
			word_hidden1[idx].setParam(&model_params._tanh1_project);
			word_hidden1[idx].init(hyper_params.hiddenSize1, hyper_params.dropProb, mem);
			word_hidden2[idx].setParam(&model_params._tanh2_project);
			word_hidden2[idx].init(hyper_params.hiddenSize2, hyper_params.dropProb, mem);
		}	
		word_window.init(hyper_params.unitSize, hyper_params.wordContext, mem);
		left_lstm.init(&model_params._left_lstm_project, hyper_params.dropProb, true, mem);
		right_lstm.init(&model_params._right_lstm_project, hyper_params.dropProb, false, mem);
		for(int idx = 0; idx < output.size(); idx++){
			outputseg[idx].init(&model_params._seglayer_project, hyper_params.dropProb, mem);
			output[idx].setParam(&model_params._olayer_linear);
			output[idx].init(hyper_params.segLabelSize, -1, mem);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		//first step: clear nodes
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		//second step: build 
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
		}

		static int offset;
		vector<PNode> segnodes;
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			segnodes.clear();
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				segnodes.push_back(&word_hidden2[idx + dist]);
				outputseg[offset + dist].forward(this, segnodes);
			}
		}
		
		poutput.resize(seq_size, max_seg_length);
		poutput = NULL;
		offset = 0;
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				output[offset + dist].forward(this, &(outputseg[offset + dist]._output));
				poutput[idx][dist] = &output[offset + dist];
			}
		}
	}

};

#endif /* SRC_ComputionGraph_H_ */

