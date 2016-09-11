/*
 * Segmentor.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#include "NNSEmbHypridCRFSegmentor.h"

#include "Argument_helper.h"

Segmentor::Segmentor() {
	// TODO Auto-generated constructor stub
}

Segmentor::~Segmentor() {
	// TODO Auto-generated destructor stub
}

int Segmentor::createAlphabet(const vector<Instance>& vecInsts) {
	if (vecInsts.size() == 0){
		std::cout << "training set empty" << std::endl;
		return -1;
	}
	cout << "Creating Alphabet..." << endl;

	int numInstance;

	m_driver._model_params._label_alpha.clear();
	m_driver._model_params._seg_label_alpha.clear();
	ignoreLabels.clear();


	int typeNum = vecInsts[0].typefeatures[0].size();
	m_type_stats.resize(typeNum);

	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->words;
		const vector<string> &labels = pInstance->labels;
		const vector<vector<string> > &sparsefeatures = pInstance->sparsefeatures;
		const vector<vector<string> > &charfeatures = pInstance->charfeatures;

		const vector<vector<string> > &typefeatures = pInstance->typefeatures;
		for (int iter_type = 0; iter_type < typefeatures.size(); iter_type++) {
			assert(typeNum == typefeatures[iter_type].size());
		}

		int curInstSize = labels.size();
		int labelId;
		for (int i = 0; i < curInstSize; ++i) {
			if (is_start_label(labels[i])){
				labelId = m_driver._model_params._seg_label_alpha.from_string(labels[i].substr(2));
			}
			else if (labels[i].length() == 1) {
				// usually O or o, trick
				labelId = m_driver._model_params._seg_label_alpha.from_string(labels[i]);
				ignoreLabels.insert(labels[i]);
			}
			labelId = m_driver._model_params._label_alpha.from_string(labels[i]);

			string curword = normalize_to_lowerwithdigit(words[i]);
			m_word_stats[curword]++;
			for (int j = 0; j < charfeatures[i].size(); j++)
				m_char_stats[charfeatures[i][j]]++;
			for (int j = 0; j < typefeatures[i].size(); j++)
				m_type_stats[j][typefeatures[i][j]]++;
			for (int j = 0; j < sparsefeatures[i].size(); j++)
				m_feat_stats[sparsefeatures[i][j]]++;
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
	cout << "Label num: " << m_driver._model_params._label_alpha.size() << endl;
	cout << "Seg Label num: " << m_driver._model_params._seg_label_alpha.size() << endl;
	m_driver._model_params._label_alpha.set_fixed_flag(true);
	m_driver._model_params._seg_label_alpha.set_fixed_flag(true);
	ignoreLabels.insert(unknownkey);

	return 0;
}

int Segmentor::addTestAlpha(const vector<Instance>& vecInsts) {
	cout << "Adding word Alphabet..." << endl;


	for (int numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->words;
		const vector<vector<string> > &charfeatures = pInstance->charfeatures;
		const vector<vector<string> > &typefeatures = pInstance->typefeatures;
		for (int iter_type = 0; iter_type < typefeatures.size(); iter_type++) {
			assert(m_type_stats.size() == typefeatures[iter_type].size());
		}
		int curInstSize = words.size();
		for (int i = 0; i < curInstSize; ++i) {
			string curword = normalize_to_lowerwithdigit(words[i]);
			if (!m_options.wordEmbFineTune)m_word_stats[curword]++;
			if (!m_options.charEmbFineTune){
				for (int j = 1; j < charfeatures[i].size(); j++){
					m_char_stats[charfeatures[i][j]]++;
				}
			}
			if (!m_options.typeEmbFineTune){
				for (int j = 0; j < typefeatures[i].size(); j++){
					m_type_stats[j][typefeatures[i][j]]++;
				}
			}
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}


	return 0;
}


void Segmentor::extractFeature(Feature& feat, const Instance* pInstance, int idx) {
	feat.clear();

	const vector<string>& words = pInstance->words;
	int sentsize = words.size();
	string curWord = idx >= 0 && idx < sentsize ? normalize_to_lowerwithdigit(words[idx]) : nullkey;

	// word features
	feat.words.push_back(curWord);
	
	// seg features
	string curSeg = "";
	for(int j = 0; j < m_options.maxsegLen; j++){
		if(idx+j < sentsize){
			curSeg = curSeg + words[idx+j];
		}
		else{
			curSeg = nullkey;
		}
		feat.segs.push_back(curSeg);
	}

	// char features

	const vector<vector<string> > &charfeatures = pInstance->charfeatures;

	const vector<string>& cur_chars = charfeatures[idx];
	for (int i = 0; i < cur_chars.size(); i++) {
		feat.chars.push_back(cur_chars[i]);
	}

	const vector<vector<string> > &typefeatures = pInstance->typefeatures;

	const vector<string>& cur_types = typefeatures[idx];
	for (int i = 0; i < cur_types.size(); i++) {
		feat.types.push_back(cur_types[i]);
	}

	const vector<string>& linear_features = pInstance->sparsefeatures[idx];
	for (int i = 0; i < linear_features.size(); i++) {
		feat.linear_features.push_back(linear_features[i]);
	}

}

void Segmentor::convert2Example(const Instance* pInstance, Example& exam, bool bTrain) {
	exam.clear();
	const vector<string> &labels = pInstance->labels;
	int curInstSize = labels.size();
	
	for (int i = 0; i < curInstSize; ++i) {
		string orcale = labels[i];

		int numLabel = m_driver._model_params._label_alpha.size();
		vector<dtype> curlabels;
		for (int j = 0; j < numLabel; ++j) {
			string str = m_driver._model_params._label_alpha.from_id(j);
			if (str.compare(orcale) == 0)
				curlabels.push_back(1.0);
			else
				curlabels.push_back(0.0);
		}

		exam.m_labels.push_back(curlabels);
		Feature feat;
		extractFeature(feat, pInstance, i);
		exam.m_features.push_back(feat);
	}

	resizeVec(exam.m_seglabels, curInstSize, m_options.maxsegLen, m_driver._model_params._seg_label_alpha.size());
	assignVec(exam.m_seglabels, 0.0);
	vector<segIndex> segs;
	getSegs(labels, segs);
	static int startIndex, disIndex, orcaleId;
	for (int idx = 0; idx < segs.size(); idx++){
		orcaleId =  m_driver._model_params._seg_label_alpha.from_string(segs[idx].label);
		startIndex = segs[idx].start;
		disIndex = segs[idx].end - segs[idx].start;
		if (disIndex < m_options.maxsegLen && orcaleId >= 0) { 
			exam.m_seglabels[startIndex][disIndex][orcaleId] = 1.0;
			if (m_driver._hyper_params.maxLabelLength[orcaleId] < disIndex + 1) 
				m_driver._hyper_params.maxLabelLength[orcaleId] = disIndex + 1;
		}
	}

	// O or o
	for (int i = 0; i < curInstSize; ++i) {
		if (labels[i].length() == 1){
			orcaleId = m_driver._model_params._seg_label_alpha.from_string(labels[i]);
			exam.m_seglabels[i][0][orcaleId] = 1.0;
			if (m_driver._hyper_params.maxLabelLength[orcaleId] < 1) 
				m_driver._hyper_params.maxLabelLength[orcaleId] = 1;
		}
	}
}

void Segmentor::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams, bool bTrain) {
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		Example curExam;
		convert2Example(pInstance, curExam, bTrain);
		vecExams.push_back(curExam);

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
}

void Segmentor::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	static vector<Instance> decodeInstResults;
	static Instance curDecodeInst;
	bool bCurIterBetter = false;

	m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

	//Ensure that each file in m_options.testFiles exists!
	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
	}

	//std::cout << "Training example number: " << trainInsts.size() << std::endl;
	//std::cout << "Dev example number: " << trainInsts.size() << std::endl;
	//std::cout << "Test example number: " << trainInsts.size() << std::endl;

	createAlphabet(trainInsts);
	addTestAlpha(devInsts);
	addTestAlpha(testInsts);
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		addTestAlpha(otherInsts[idx]);
	}

	vector<Example> trainExamples, devExamples, testExamples, allExamples;
	m_driver._hyper_params.maxsegLen = m_options.maxsegLen;
	m_driver._hyper_params.maxLabelLength.resize(m_driver._model_params._seg_label_alpha.size());
	assignVec(m_driver._hyper_params.maxLabelLength, 0);
	initialExamples(trainInsts, trainExamples, true);
	//print length information
	std::cout << "Predefined max seg length: " << m_options.maxsegLen << std::endl;
	for (int j = 0; j < m_driver._model_params._seg_label_alpha.size(); j++){
		std::cout << "max length of label " << m_driver._model_params._seg_label_alpha.from_id(j) << ": " << m_driver._hyper_params.maxLabelLength[j] << std::endl;
	}
	m_driver._model_params._loss.initial(m_driver._hyper_params.maxLabelLength, m_options.maxsegLen);
	initialExamples(devInsts, devExamples);
	initialExamples(testInsts, testExamples);

	addAllItems(allExamples, trainExamples);
	addAllItems(allExamples, devExamples);
	addAllItems(allExamples, testExamples);

	vector<int> otherInstNums(otherInsts.size());
	vector<vector<Example> > otherExamples(otherInsts.size());
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		initialExamples(otherInsts[idx], otherExamples[idx]);
		otherInstNums[idx] = otherExamples[idx].size();
		addAllItems(allExamples, otherExamples[idx]);
	}

	m_word_stats[unknownkey] = m_options.wordCutOff + 1;
	m_driver._model_params._word_alpha.initial(m_word_stats, m_options.wordCutOff);
	if (m_options.wordFile != "") {
		m_driver._model_params._words.initial(&m_driver._model_params._word_alpha, m_options.wordFile, m_options.wordEmbFineTune);
	}
	else{
		m_driver._model_params._words.initial(&m_driver._model_params._word_alpha, m_options.wordEmbSize, m_options.wordEmbFineTune);
	}

	int typeNum = m_type_stats.size();
	m_driver._model_params._types.resize(typeNum);
	m_driver._model_params._type_alphas.resize(typeNum);
	
	for (int idx = 0; idx < typeNum; idx++){
		m_type_stats[idx][unknownkey] = 1; // use the s
		m_driver._model_params._type_alphas[idx].initial(m_type_stats[idx]);
		if (m_options.typeFiles.size() > idx && m_options.typeFiles[idx] != "") {
			m_driver._model_params._types[idx].initial(&m_driver._model_params._type_alphas[idx], m_options.typeFiles[idx], m_options.typeEmbFineTune);
		}
		else{
			m_driver._model_params._types[idx].initial(&m_driver._model_params._type_alphas[idx], m_options.typeEmbSize, m_options.typeEmbFineTune);
		}
	}

	collectSEGAlpha(allExamples, m_options.segFile);

	// use rnnHiddenSize to replace segHiddensize
	m_driver._hyper_params.setRequared(m_options);
	m_driver.initial();


	dtype bestDIS = 0;

	int inputSize = trainExamples.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test, metric_dev2, metric_test2;
	static vector<Example> subExamples;
	int devNum = devExamples.size(), testNum = testExamples.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;

		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
			subExamples.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			for (int idy = start_pos; idy < end_pos; idy++) {
				subExamples.push_back(trainExamples[indexes[idy]]);
			}

			int curUpdateIter = iter * batchBlock + updateIter;
			dtype cost = m_driver.train(subExamples, curUpdateIter);

			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
				//m_driver.checkgrad(subExamples, curUpdateIter + 1);
				std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
			}
			m_driver.updateModel();

		}

		if (devNum > 0) {
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			metric_dev2.reset();
			for (int idx = 0; idx < devExamples.size(); idx++) {
				vector<string> result_labels, result_label2s;
				predict(devExamples[idx].m_features, result_labels, result_label2s);

				if (m_options.seg){
					devInsts[idx].SegEvaluate(result_labels, metric_dev);
					devInsts[idx].SegEvaluate(result_label2s, metric_dev2);
				}
				else{
					devInsts[idx].Evaluate(result_labels, metric_dev);
					devInsts[idx].Evaluate(result_label2s, metric_dev2);
				}

				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_labels);
					curDecodeInst.assignAdditionLabel(result_label2s);
					decodeInstResults.push_back(curDecodeInst);
				}
			}

			metric_dev.print();
			metric_dev2.print();

			if (!m_options.outBest.empty() && (metric_dev.getAccuracy() + metric_dev2.getAccuracy()) / 2.0 > bestDIS) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				metric_test2.reset();
				for (int idx = 0; idx < testExamples.size(); idx++) {
					vector<string> result_labels, result_label2s;
					predict(testExamples[idx].m_features, result_labels, result_label2s);

					if (m_options.seg){
						testInsts[idx].SegEvaluate(result_labels, metric_test);
						testInsts[idx].SegEvaluate(result_label2s, metric_test2);
					}
					else{
						testInsts[idx].Evaluate(result_labels, metric_test);
						testInsts[idx].Evaluate(result_label2s, metric_test2);
					}

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_labels);
						curDecodeInst.assignAdditionLabel(result_label2s);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();
				metric_test2.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherExamples.size(); idx++) {
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				metric_test2.reset();
				for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
					vector<string> result_labels, result_label2s;
					predict(otherExamples[idx][idy].m_features, result_labels, result_label2s);

					if (m_options.seg){
						otherInsts[idx][idy].SegEvaluate(result_labels, metric_test);
						otherInsts[idx][idy].SegEvaluate(result_label2s, metric_test2);
					}
					else{
						otherInsts[idx][idy].Evaluate(result_labels, metric_test);
						otherInsts[idx][idy].Evaluate(result_label2s, metric_test2);
					}

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
						curDecodeInst.assignLabel(result_labels);
						curDecodeInst.assignAdditionLabel(result_label2s);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();
				metric_test2.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}

			if (m_options.saveIntermediate && (metric_dev.getAccuracy() + metric_dev2.getAccuracy()) / 2.0 > bestDIS) {
				std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
				bestDIS = (metric_dev.getAccuracy() + metric_dev2.getAccuracy())/ 2.0 ;
				writeModelFile(modelFile);
			}

		}
		// Clear gradients
	}
}

int Segmentor::predict(const vector<Feature>& features, vector<string>& outputs, vector<string>& output2s) {
	//assert(features.size() == words.size());
	NRMat<int> labelIdx;
	vector<int> labelIdx2;
	m_driver.predict(features, labelIdx, labelIdx2);
	int seq_size = features.size();
	outputs.resize(seq_size);
	for (int idx = 0; idx < seq_size; idx++) {
		outputs[idx] = nullkey;
	}
	for (int idx = 0; idx < seq_size; idx++) {
		for (int dist = 0; idx + dist < seq_size && dist < m_options.maxsegLen; dist++) {
			if (labelIdx[idx][dist] < 0) continue;
			string label = m_driver._model_params._seg_label_alpha.from_id(labelIdx[idx][dist], unknownkey);
			for (int i = idx; i <= idx + dist; i++){
				if (outputs[i] != nullkey) {
					std::cout << "predict error" << std::endl;
				}
			}
			if (ignoreLabels.find(label) != ignoreLabels.end()){
				for (int i = idx; i <= idx + dist; i++){
					outputs[i] = label;
				}
			}
			else{
				if (dist == 0){
					outputs[idx] = "s-" + label;
				}
				else{
					outputs[idx] = "b-" + label;
					for (int i = idx + 1; i < idx + dist; i++){
						outputs[i] = "m-" + label;
					}
					outputs[idx + dist] = "e-" + label;
				}
			}			
		}
	}

	for (int idx = 0; idx < seq_size; idx++) {
		if (outputs[idx] == nullkey){
			std::cout << "predict error" << std::endl;
		}
	}

	output2s.resize(seq_size);
	for (int idx = 0; idx < seq_size; idx++) {
		output2s[idx] = m_driver._model_params._label_alpha.from_id(labelIdx2[idx], unknownkey);
	}

	for (int idx = 0; idx < seq_size; idx++) {
		if (output2s[idx] == nullkey){
			std::cout << "predict error" << std::endl;
		}
	}

	return 0;
}

void Segmentor::test(const string& testFile, const string& outputFile, const string& modelFile) {
	loadModelFile(modelFile);
	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts);

	vector<Example> testExamples;
	initialExamples(testInsts, testExamples);

	int testNum = testExamples.size();
	vector<Instance> testInstResults;
	Metric metric_test, metric_test2;
	metric_test.reset();
	metric_test2.reset();
	for (int idx = 0; idx < testExamples.size(); idx++) {
		vector<string> result_labels, result_label2s;
		predict(testExamples[idx].m_features, result_labels, result_label2s);
		if (m_options.seg){
			testInsts[idx].SegEvaluate(result_labels, metric_test);
			testInsts[idx].SegEvaluate(result_label2s, metric_test2);
		}
		else{
			testInsts[idx].Evaluate(result_labels, metric_test);
			testInsts[idx].Evaluate(result_label2s, metric_test2);			
		}
		Instance curResultInst;
		curResultInst.copyValuesFrom(testInsts[idx]);
		curResultInst.assignLabel(result_labels);
		curResultInst.assignAdditionLabel(result_label2s);
		testInstResults.push_back(curResultInst);
	}
	std::cout << "test:" << std::endl;
	metric_test.print();
	metric_test2.print();

	m_pipe.outputAllInstances(outputFile, testInstResults);

}

void Segmentor::collectSEGAlpha(const vector<Example>& vecInsts, const string& segFile){
	hash_map<string, int> seg_stats;
	for (int idx = 0; idx < vecInsts.size(); idx++){
		for (int idy = 0; idy < vecInsts[idx].m_features.size(); idy++){
			const vector<string>& segs = vecInsts[idx].m_features[idy].segs;
			for (int idz = 0; idz < segs.size(); idz++){
				seg_stats[segs[idz]]++;
			}
		}
	}
	
	static ifstream inf;
	if (inf.is_open()) {
		inf.close();
		inf.clear();
	}
	inf.open(segFile.c_str());

	static string strLine;
	static vector<string> vecInfo;
	while (1) {
		if (!my_getline(inf, strLine)) {
			break;
		}
		if (strLine.empty()){
			continue;
		}
		split_bychar(strLine, vecInfo, ' ');
		if (seg_stats.find(vecInfo[0]) != seg_stats.end()){
			m_seg_stats[vecInfo[0]]++;
		}
	}
	inf.close();

	std::cout << "all possible segs: " << seg_stats.size() << ", hited segs: " << m_seg_stats.size() 
		<< ", OOV ratio = " << (seg_stats.size() - m_seg_stats.size()) * 1.0 / seg_stats.size() << std::endl;
	m_driver._model_params._seg_alpha.initial(m_seg_stats);
	m_driver._model_params._segs.initial(&m_driver._model_params._seg_alpha, segFile, false);
}

void Segmentor::loadModelFile(const string& inputModelFile) {

}

void Segmentor::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

	ah.process(argc, argv);

	Segmentor segmentor;
	segmentor.m_pipe.max_sentense_size = ComputionGraph::max_sentence_length;
	if (bTrain) {
		segmentor.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		segmentor.test(testFile, outputFile, modelFile);
	}

	//test(argv);
	//ah.write_values(std::cout);
}
