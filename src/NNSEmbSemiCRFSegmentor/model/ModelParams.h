#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"
#include "BMESSegmentation.h"

class ModelParams{
public:

	LSTM1Params _left_lstm_project; //left lstm
	LSTM1Params _right_lstm_project; //right lstm
	UniParams _tanh1_project; // hidden
	BiParams _tanh2_project; // hidden
	SegParams _seglayer_project; //segmentation
	BiParams _segtanh_project; //
	UniParams _olayer_linear; // output

	Semi0CRFMLLoss _loss;

public://follow parameters should be initialized outside
	vector<LookupTable> _types;
	LookupTable _words;
	LookupTable _segs;


	vector<Alphabet> _type_alphas;	
	Alphabet _word_alpha;
	Alphabet _label_alpha;
	Alphabet _seg_label_alpha;
	Alphabet _seg_alpha;

public:
	bool initial(HyperParams& hyper_params){
		if (_words.nVSize <= 0 || _label_alpha.size() < 0 || _segs.nVSize < 0)
			return false;
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		hyper_params.wordDim = _words.nDim;
		hyper_params.segDim = _segs.nDim;
		hyper_params.unitSize = hyper_params.wordDim;
		hyper_params.typeDims.clear();
		for (int idx = 0; idx < _types.size(); idx++)
		{
			if (_types[idx].nVSize <= 0 || _type_alphas[idx].size() <= 0)
				return false;
			hyper_params.typeDims.push_back(_types[idx].nDim);
			hyper_params.unitSize += hyper_params.typeDims[idx];
		}
		hyper_params.segLabelSize = _seg_label_alpha.size();
		hyper_params.inputSize = hyper_params.wordWindow * hyper_params.unitSize;

		_tanh1_project.initial(hyper_params.hiddenSize1, hyper_params.inputSize, true);
		_left_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize1);
		_right_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize1);
		_tanh2_project.initial(hyper_params.hiddenSize1, hyper_params.rnnHiddenSize, hyper_params.rnnHiddenSize, true);
		_seglayer_project.initial(hyper_params.segHiddenSize, hyper_params.hiddenSize2, hyper_params.hiddenSize1);
		_segtanh_project.initial(hyper_params.segHiddenSize, hyper_params.segHiddenSize, hyper_params.segDim, true);
		_olayer_linear.initial(hyper_params.segLabelSize, hyper_params.segHiddenSize, false);
	
		_loss.initial(hyper_params.maxLabelLength, hyper_params.maxsegLen);

		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		_words.exportAdaParams(ada);
		for (int idx = 0; idx < _types.size(); idx++)
			_types[idx].exportAdaParams(ada);
		_tanh1_project.exportAdaParams(ada);
		_left_lstm_project.exportAdaParams(ada);
		_right_lstm_project.exportAdaParams(ada);
		_tanh2_project.exportAdaParams(ada);
		_seglayer_project.exportAdaParams(ada);
		_segtanh_project.exportAdaParams(ada);
		_olayer_linear.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&_tanh1_project.W, "_tanh1_project.W");
		checkgrad.add(&_tanh1_project.b, "_tanh1_project.b");
		checkgrad.add(&_tanh2_project.W1, "_tanh2_project.W1");
		checkgrad.add(&_tanh2_project.W2, "_tanh2_project.W2");
		checkgrad.add(&_tanh2_project.b, "_tanh2_project.b");
		checkgrad.add(&_seglayer_project.B.W, "_seglayer_project.B.W");
		checkgrad.add(&_seglayer_project.M.W, "_seglayer_project.M.W");
		checkgrad.add(&_seglayer_project.E.W, "_seglayer_project.E.W");;
		checkgrad.add(&_seglayer_project.S.W, "_seglayer_project.S.W");;

	}

	void saveModel(){
	}

	void loadModel(const string& inFile){
	}
};


#endif /*SRC_ModelParams_H_ */
