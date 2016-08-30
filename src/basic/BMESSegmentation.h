/*
 * SegOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef BMESSEGBuilder_H_
#define BMESSEGBuilder_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Pooling.h"
#include "UniOP.h"
#include "FourOP.h"

struct SegParams {
	UniParams B;
	UniParams M;
	UniParams E;
	UniParams S;
	FourParams merge;
	LookupTable lengths;
	int inDim;
	int outDim;
	int hiddenDim;
	int maxLength;
	int lengthDim;

	SegParams() {	
		maxLength = 5;  //  fixed
		lengthDim = 20; //  fixed
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		B.exportAdaParams(ada);
		M.exportAdaParams(ada);
		E.exportAdaParams(ada);
		S.exportAdaParams(ada);
		merge.exportAdaParams(ada);
		lengths.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nHSize, int nISize, int seed = 0) {
		B.initial(nHSize, nISize, true, seed);
		M.initial(nHSize, nISize, true, seed + 1);
		E.initial(nHSize, nISize, true, seed + 2);
		S.initial(nHSize, nISize, true, seed + 3);
		merge.initial(nOSize, nHSize, nHSize, nHSize, lengthDim, true, seed + 4);
		inDim = nISize;
		outDim = nOSize;
		hiddenDim = nHSize;	
		hash_map<string, int> length_stat;
		for (int idx = 1; idx <= maxLength; idx++){
			length_stat[obj2string(idx)] = 1;
		}
		lengths.initial(length_stat, 0, lengthDim, seed + 5, true);
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

	FourNode _output;
	SumPoolNode _sum;
	MaxPoolNode _max;
	MinPoolNode _min;
	LookupNode _length;
	vector<UniNode> _tnodes;

	DropNode _length_drop;
	vector<DropNode> _tnodes_drop;
	DropNode _output_drop;
	

public:
	SegBuilder(){
		clear();
	}

	~SegBuilder(){
		clear();
	}

	inline void setParam(SegParams* paramInit, dtype dropout) {
		_param = paramInit;
		_inDim = _param->inDim;
		_outDim = _param->outDim;
		_hiddenDim = _param->hiddenDim;
		_length.setParam(&_param->lengths);
		_output.setParam(&_param->merge);

		for (int idx = 0; idx < _tnodes.size(); idx++){
			_tnodes_drop[idx].setDropValue(dropout);
		}
		_length_drop.setDropValue(dropout);
		_output_drop.setDropValue(dropout);
	}

	inline void setFunctions(Mat(*f)(const Mat&),
		Mat(*f_deri)(const Mat&, const Mat&)) {
		for (int idx = 0; idx < _tnodes.size(); idx++){
			_tnodes[idx].setFunctions(f, f_deri);
		}
		_output.setFunctions(f, f_deri);
	}

	inline void resize(int maxsize){
		_tnodes.resize(maxsize);
		_tnodes_drop.resize(maxsize);
	}

	inline void clear(){
		_tnodes.clear();
		_tnodes_drop.clear();
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

		if (_nSize == 1){
			_tnodes[0].setParam(&(_param->S));
		}
		else{
			_tnodes[0].setParam(&(_param->B));
			for (int idx = 1; idx < _nSize - 1; idx++){
				_tnodes[idx].setParam(&(_param->M));
			}
			_tnodes[_nSize-1].setParam(&(_param->E));
		}

		for (int idx = 0; idx < _nSize; idx++){
			_tnodes[idx].forward(cg, x[idx]);
			_tnodes_drop[idx].forward(cg, &_tnodes[idx], bTrain);
		}

		_sum.forward(cg, getPNodes(_tnodes_drop, _nSize));
		_max.forward(cg, getPNodes(_tnodes_drop, _nSize));
		_min.forward(cg, getPNodes(_tnodes_drop, _nSize));

		_length.forward(cg, obj2string(_nSize < _param->maxLength ? _nSize : _param->maxLength));
		_length_drop.forward(cg, &_length, bTrain);

		_output.forward(cg, &_sum, &_max, &_min, &_length_drop);
		_output_drop.forward(cg, &_output, bTrain);
	}

};

#endif /* BMESSEGBuilder_H_ */
