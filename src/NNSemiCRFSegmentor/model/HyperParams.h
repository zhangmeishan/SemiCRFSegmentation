#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
public:
	int wordContext, wordWindow;
	int wordDim;
	int maxsegLen;
	vector<int> typeDims;
	vector<int> maxLabelLength;
	int unitSize;
	dtype dropProb;


	int hiddenSize1;
	int rnnHiddenSize;
	int hiddenSize2;
	int segHiddenSize;
	int inputSize;
	int labelSize;

	dtype nnRegular, adaAlpha, adaEps;

public:
	HyperParams(){
		bAssigned = false;
	}

	void setRequared(Options& opt){
		wordContext = opt.wordcontext;
		typeDims.clear();
		unitSize = wordDim;
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

	void clear() {
		bAssigned = false;
	}

	bool bVaild(){
		return bAssigned;
	}

	void print(){

	}
private:
	bool bAssigned;
};

#endif /* SRC_HyperParams_H_*/
