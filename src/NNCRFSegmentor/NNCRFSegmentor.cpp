/*
 * Segmentor.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#include "NNCRFSegmentor.h"

#include "Argument_helper.h"

Segmentor::Segmentor(int memsize) : m_driver(memsize){
	// TODO Auto-generated constructor stub
	srand(0);
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

	m_driver._modelparams.labelAlpha.clear();

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
			labelId = m_driver._modelparams.labelAlpha.from_string(labels[i]);

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
	cout << "Label num: " << m_driver._modelparams.labelAlpha.size() << endl;
	m_driver._modelparams.labelAlpha.set_fixed_flag(true);

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

void Segmentor::convert2Example(const Instance* pInstance, Example& exam) {
	exam.clear();
	const vector<string> &labels = pInstance->labels;
	int curInstSize = labels.size();
	
	for (int i = 0; i < curInstSize; ++i) {
		string orcale = labels[i];

		int numLabel = m_driver._modelparams.labelAlpha.size();
		vector<dtype> curlabels;
		for (int j = 0; j < numLabel; ++j) {
			string str = m_driver._modelparams.labelAlpha.from_id(j);
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

}

void Segmentor::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		Example curExam;
		convert2Example(pInstance, curExam);
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

	vector<Example> trainExamples, devExamples, testExamples;

	initialExamples(trainInsts, trainExamples);
	initialExamples(devInsts, devExamples);
	initialExamples(testInsts, testExamples);

	vector<int> otherInstNums(otherInsts.size());
	vector<vector<Example> > otherExamples(otherInsts.size());
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		initialExamples(otherInsts[idx], otherExamples[idx]);
		otherInstNums[idx] = otherExamples[idx].size();
	}

	m_word_stats[unknownkey] = m_options.wordCutOff + 1;
	m_driver._modelparams.wordAlpha.initial(m_word_stats, m_options.wordCutOff);
	if (m_options.wordFile != "") {
		m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordFile, m_options.wordEmbFineTune);
	}
	else{
		m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordEmbSize, m_options.wordEmbFineTune);
	}

	int typeNum = m_type_stats.size();
	m_driver._modelparams.typeAlphas.resize(typeNum);
	m_driver._modelparams.types.resize(typeNum);
	for (int idx = 0; idx < typeNum; idx++){
		m_type_stats[idx][unknownkey] = 1; // use the s
		m_driver._modelparams.typeAlphas[idx].initial(m_type_stats[idx], 0);
		if (m_options.typeFiles.size() > idx && m_options.typeFiles[idx] != "") {
			m_driver._modelparams.types[idx].initial(&(m_driver._modelparams.typeAlphas[idx]), m_options.typeFiles[idx], m_options.typeEmbFineTune);
		}
		else{
			m_driver._modelparams.types[idx].initial(&(m_driver._modelparams.typeAlphas[idx]), m_options.typeEmbSize, m_options.typeEmbFineTune);
		}
	}

	
	m_driver._hyperparams.setRequared(m_options);
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

	static Metric eval, metric_dev, metric_test;
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
				m_driver.checkgrad(subExamples, curUpdateIter + 1);
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
			for (int idx = 0; idx < devExamples.size(); idx++) {
				vector<string> result_labels;
				predict(devExamples[idx].m_features, result_labels);

				if (m_options.seg)
					devInsts[idx].SegEvaluate(result_labels, metric_dev);
				else
					devInsts[idx].Evaluate(result_labels, metric_dev);

				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_labels);
					decodeInstResults.push_back(curDecodeInst);
				}
			}

			metric_dev.print();

			if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idx = 0; idx < testExamples.size(); idx++) {
					vector<string> result_labels;
					predict(testExamples[idx].m_features, result_labels);

					if (m_options.seg)
						testInsts[idx].SegEvaluate(result_labels, metric_test);
					else
						testInsts[idx].Evaluate(result_labels, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_labels);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherExamples.size(); idx++) {
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
					vector<string> result_labels;
					predict(otherExamples[idx][idy].m_features, result_labels);

					if (m_options.seg)
						otherInsts[idx][idy].SegEvaluate(result_labels, metric_test);
					else
						otherInsts[idx][idy].Evaluate(result_labels, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
						curDecodeInst.assignLabel(result_labels);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}

			if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS) {
				std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
				bestDIS = metric_dev.getAccuracy();
				writeModelFile(modelFile);
			}

		}
		// Clear gradients
	}
}

int Segmentor::predict(const vector<Feature>& features, vector<string>& outputs) {
	//assert(features.size() == words.size());
	vector<int> labelIdx;
	m_driver.predict(features, labelIdx);
	int seq_size = features.size();
	outputs.resize(seq_size);
	for (int idx = 0; idx < seq_size; idx++) {
		outputs[idx] = m_driver._modelparams.labelAlpha.from_id(labelIdx[idx], unknownkey);
	}

	for (int idx = 0; idx < seq_size; idx++) {
		if (outputs[idx] == nullkey){
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
	Metric metric_test;
	metric_test.reset();
	for (int idx = 0; idx < testExamples.size(); idx++) {
		vector<string> result_labels;
		predict(testExamples[idx].m_features, result_labels);
		if (m_options.seg)
			testInsts[idx].SegEvaluate(result_labels, metric_test);
		else
			testInsts[idx].Evaluate(result_labels, metric_test);		
		Instance curResultInst;
		curResultInst.copyValuesFrom(testInsts[idx]);
		curResultInst.assignLabel(result_labels);
		testInstResults.push_back(curResultInst);
	}
	std::cout << "test:" << std::endl;
	metric_test.print();

	m_pipe.outputAllInstances(outputFile, testInstResults);

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
	int memsize = 0;


	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_int("memsize", "memorySize", "named_int", "This argument decides the size of static memory allocation", memsize);

	ah.process(argc, argv);
	
	if (memsize < 0)
		memsize = 0;
	Segmentor segmentor(memsize);
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
