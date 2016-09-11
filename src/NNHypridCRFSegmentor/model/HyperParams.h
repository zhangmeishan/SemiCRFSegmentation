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
	int rnnHiddenSize;
	int segHiddenSize;
	int maxsegLen;
	dtype dropProb;

	//auto generated
	int wordWindow;
	int wordDim;
	vector<int> typeDims;
	vector<int> maxLabelLength;
	int unitSize;
	int inputSize;
	int labelSize;
	int segLabelSize;


	// for optimization
	dtype nnRegular, adaAlpha, adaEps;
public:
	HyperParams(){
		bAssigned = false;
	}

	bool bVaild(){
		return bAssigned;
	}

	void setRequared(Options& opt){
		wordContext = opt.wordcontext;
		hiddenSize1 = opt.hiddenSize;
		hiddenSize2 = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
		segHiddenSize = opt.segHiddenSize;
		maxsegLen = opt.maxsegLen;
		dropProb = opt.dropProb;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}
	void print(){}
private:
	bool bAssigned;
};

#endif /*SRC_HyperParams_H_*/