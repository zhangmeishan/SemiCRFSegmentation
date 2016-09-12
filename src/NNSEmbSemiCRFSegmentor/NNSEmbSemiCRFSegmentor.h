/*
 * Segmentor.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_NNSEmbSemiCRFSegmentor_H_
#define SRC_NNSEmbSemiCRFSegmentor_H_


#include "N3L.h"
#include "Driver.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Segmentor {


public:
	unordered_set<string> ignoreLabels;
	unordered_map<string, int> m_feat_stats;
	unordered_map<string, int> m_word_stats;
	unordered_map<string, int> m_char_stats;
	vector<unordered_map<string, int> > m_type_stats;
	unordered_map<string, int> m_seg_stats; // read it by file

public:
	Options m_options;
	Driver m_driver;
	Pipe m_pipe;


public:
	Segmentor();
	virtual ~Segmentor();

public:

	int createAlphabet(const vector<Instance>& vecTrainInsts);
	int addTestAlpha(const vector<Instance>& vecInsts);
	void collectSEGAlpha(const vector<Example>& vecInsts, const string& segFile); //notice: seg embeddings are fixed during training


	void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
	void extractFeature(Feature& feat, const Instance* pInstance, int idx);

	void convert2Example(const Instance* pInstance, Example& exam);
	void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
	int predict(const vector<Feature>& features, vector<string>& outputs);
	void test(const string& testFile, const string& outputFile, const string& modelFile);

	void writeModelFile(const string& outputModelFile);
	void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_NNSEmbSemiCRFSegmentor_H_ */
