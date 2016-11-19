/*
 * SegOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef _BMESSEGBuilder_H_
#define _BMESSEGBuilder_H_

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

	Alphabet length_alpha; // for lengths initial.
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

	inline void initial(int nOSize, int nHSize, int nISize, AlignedMemoryPool* mem = NULL){
		B.initial(nHSize, nISize, true, mem);
		M.initial(nHSize, nISize, true, mem);
		E.initial(nHSize, nISize, true, mem);
		S.initial(nHSize, nISize, true, mem);
		merge.initial(nOSize, nHSize, nHSize, nHSize, lengthDim, true, mem);
		inDim = nISize;
		outDim = nOSize;
		hiddenDim = nHSize;	
		unordered_map<string, int> length_stat;

		for (int idx = 1; idx <= maxLength; idx++){
			length_stat[obj2string(idx)] = 1;
		}
		length_alpha.initial(length_stat);
		lengths.initial(&length_alpha, lengthDim, true);
	}
};

// we can rewrite it as one node, but many duplicated codes
class SegBuilder{
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
public:
	SegBuilder(){
		clear();
	}

	~SegBuilder(){
		clear();
	}

	inline void init(SegParams* paramInit, dtype dropout, AlignedMemoryPool* mem = NULL) {
		_param = paramInit;
		_inDim = _param->inDim;
		_outDim = _param->outDim;
		_hiddenDim = _param->hiddenDim;
		_length.setParam(&_param->lengths);
		_length.init(_param->lengthDim, dropout, mem);
		_sum.setParam(_hiddenDim);
		_sum.init(_hiddenDim, -1, mem);
		_max.setParam(_hiddenDim);
		_max.init(_hiddenDim, -1, mem);
		_min.setParam(_hiddenDim);
		_min.init(_hiddenDim, -1, mem);
		int	maxsize = _tnodes.size();
		for (int idx = 0; idx < maxsize; idx++)
			_tnodes[idx].init(_hiddenDim, dropout, mem);
		_output.setParam(&_param->merge);
		_output.init(_outDim, dropout, mem);
	}

	inline void setFunctions(dtype(*f)(const dtype&),
		dtype(*f_deri)(const dtype&, const dtype&)) {
		for (int idx = 0; idx < _tnodes.size(); idx++){
			_tnodes[idx].setFunctions(f, f_deri);
		}
		_output.setFunctions(f, f_deri);
	}

	inline void resize(int maxsize){
		_tnodes.resize(maxsize);
	}

	inline void clear(){
		_tnodes.clear();
		_param = NULL;
		_inDim = 0;
		_outDim = 0;
		_hiddenDim = 0;
		_nSize = 0;
	}

public:

	inline void forward(Graph *cg, const vector<PNode>& x){
		if (x.size() == 0){
			std::cout << "empty inputs for seg operation" << std::endl;
			return;
		}

		_nSize = x.size();
		if (x[0]->val.dim != _inDim){
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
		}

		_sum.forward(cg, getPNodes(_tnodes, _nSize));
		_max.forward(cg, getPNodes(_tnodes, _nSize));
		_min.forward(cg, getPNodes(_tnodes, _nSize));

		_length.forward(cg, obj2string(_nSize < _param->maxLength ? _nSize : _param->maxLength));

		_output.forward(cg, &_sum, &_max, &_min, &_length);
	}

};

#endif /* BMESSEGBuilder_H_ */
