/*
 * SegOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef SEGBuilder_H_
#define SEGBuilder_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Pooling.h"
#include "UniOP.h"
#include "TriOP.h"

struct SegParams {
	UniParams H;
	TriParams merge;
	int inDim;
	int outDim;
	int hiddenDim;

	SegParams() {		
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		H.exportAdaParams(ada);
		merge.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nHSize, int nISize, int seed = 0) {
		H.initial(nHSize, nISize, true, seed);
		merge.initial(nOSize, nHSize, nHSize, nHSize, true, seed + 1);
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

	TriNode _output;
	SumPoolNode _sum;
	MaxPoolNode _max;
	MinPoolNode _min;
	vector<UniNode> _tnodes;
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
		_output.setParam(&_param->merge);

		for (int idx = 0; idx < _tnodes.size(); idx++){
			_tnodes[idx].setParam(&(_param->H));
			_tnodes_drop[idx].setDropValue(dropout);			
		}
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

		for (int idx = 0; idx < _nSize; idx++){
			_tnodes[idx].forward(cg, x[idx]);
			_tnodes_drop[idx].forward(cg, &_tnodes[idx], bTrain);
		}

		_sum.forward(cg, getPNodes(_tnodes_drop, _nSize));
		_max.forward(cg, getPNodes(_tnodes_drop, _nSize));
		_min.forward(cg, getPNodes(_tnodes_drop, _nSize));
		_output.forward(cg, &_sum, &_max, &_min);
		_output_drop.forward(cg, &_output, bTrain);
	}

};

#endif /* SEGBuilder_H_ */
