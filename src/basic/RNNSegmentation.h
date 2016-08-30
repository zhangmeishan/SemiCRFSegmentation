/*
 * SegOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef RNNSEGBuilder_H_
#define RNNSEGBuilder_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Pooling.h"
#include "UniOP.h"

struct SegParams {
	LSTM1Params L;
	LSTM1Params R;
	BiParams LR;
	int inDim;
	int outDim;
	int hiddenDim;

	SegParams() {
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		L.exportAdaParams(ada);
		R.exportAdaParams(ada);
		LR.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nHSize, int nISize, int seed = 0) {
		L.initial(nHSize, nISize, seed);
		R.initial(nHSize, nISize, seed + 10);
		LR.initial(nOSize, nHSize, nHSize, seed + 10);
		inDim = nISize;
		outDim = nOSize;
		hiddenDim = nHSize;
	}
};

// we can rewrite it as one node, but many duplicated codes
class SegBuilder : NodeBuilder{
public:
	SegParams* _param;

	int _nSize;
	int _inDim;
	int _outDim;
	int _hiddenDim;

	LSTM1Builder _left_lstm;
	LSTM1Builder _right_lstm;
	BiNode _output;
	DropNode _output_drop;


public:
	SegBuilder(){
		clear();
	}

	~SegBuilder(){
		clear();
	}

	inline void setParam(SegParams* paramInit, dtype dropout) {
		if (_right_lstm.empty()){
			std::cout << "please call resize() function first" << std::endl;
		}
		_param = paramInit;
		_inDim = _param->inDim;
		_outDim = _param->outDim;
		_hiddenDim = _param->hiddenDim;

		_output.setParam(&_param->LR);
		_left_lstm.setParam(&_param->L, dropout, true);
		_right_lstm.setParam(&_param->R, dropout, false);
		_output_drop.setDropValue(dropout);
	}

	inline void setFunctions(Mat(*f)(const Mat&),
		Mat(*f_deri)(const Mat&, const Mat&)) {
		_output.setFunctions(f, f_deri);
	}

	inline void resize(int maxsize){
		_left_lstm.resize(maxsize);
		_right_lstm.resize(maxsize);
	}

	inline void clear(){
		_output.clear();
		_left_lstm.clear();
		_right_lstm.clear();
		_param = NULL;
		_inDim = 0;
		_outDim = 0;
		_hiddenDim = 0;
		_nSize = 0;
	}

public:

	inline void forward(Graph *cg, const vector<PNode>& x, bool bTrain){
		if (x.size() == 0){
			std::cout << "empty inputs for seg operation" << std::endl;
			return;
		}

		_nSize = x.size();
		if (x[0]->val.rows() != _inDim){
			std::cout << "input dim does not match for seg operation" << std::endl;
			return;
		}

		_left_lstm.forward(cg, x, bTrain);
		_right_lstm.forward(cg, x, bTrain);
		_output.forward(cg, &_left_lstm._hiddens_drop[_nSize - 1], &_right_lstm._hiddens_drop[0]);
		_output_drop.forward(cg, &_output, bTrain);
	}

};

#endif /* RNNSEGBuilder_H_ */
