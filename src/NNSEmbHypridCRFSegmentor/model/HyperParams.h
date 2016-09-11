#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
public:
	// must assign
	int wordContext;
	int hiddenSize1;
	int hiddenSize2;
	int maxsegLen;
	int rnnHiddenSize;
	int segHiddenSize;
	dtype dropProb;

	// auto generated
	int wordWindow;
	int wordDim;
	vector<int> typeDims;
	vector<int> maxLabelLength;
	int unitSize;
	int segDim;
	int inputSize;
	int labelSize;
	int segLabelSize;


	// for optimization
	dtype nnRegular, adaAlpha, adaEps;
public:
	HyperParams(){
		bAssigned = false;
	}

	void setRequared(Options& opt){
		wordContext = opt.wordcontext;
		hiddenSize1 = opt.hiddenSize;
		hiddenSize2 = opt.hiddenSize;
		maxsegLen = opt.maxsegLen;
		dropProb = opt.dropProb;
		rnnHiddenSize = opt.rnnHiddenSize;
		segHiddenSize = opt.segHiddenSize;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		bAssigned = true;
	}

	bool bVaild(){
		return bAssigned;
	}

	void clear(){
		bAssigned = false;
	}

	void print(){
	
	}

private:
	bool bAssigned;
};

#endif /*SRC_HyperParams_H_*/