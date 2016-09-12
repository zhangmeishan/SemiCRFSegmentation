#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "N3L.h"
#include "Metric.h"

using namespace std;

class Instance {
public:
	Instance() {
		useAddition = false;
	}
	~Instance() {
	}

	int size() const {
		return words.size();
	}

	void clear() {
		useAddition = false;
		labels.clear();
		additionlabels.clear();
		words.clear();
		for (int i = 0; i < size(); i++) {
			sparsefeatures[i].clear();
			charfeatures[i].clear();
			typefeatures[i].clear();
		}
		sparsefeatures.clear();
		charfeatures.clear();
		typefeatures.clear();
	}

	void allocate(int length) {
		clear();
		labels.resize(length);
		additionlabels.resize(length);
		words.resize(length);
		typefeatures.resize(length);
		sparsefeatures.resize(length);
		charfeatures.resize(length);
	}

	void copyValuesFrom(const Instance& anInstance) {
		allocate(anInstance.size());
		for (int i = 0; i < anInstance.size(); i++) {
			labels[i] = anInstance.labels[i];
			additionlabels[i] = anInstance.additionlabels[i];
			words[i] = anInstance.words[i];
			for (int j = 0; j < anInstance.sparsefeatures[i].size(); j++) {
				sparsefeatures[i].push_back(anInstance.sparsefeatures[i][j]);
			}
			for (int j = 0; j < anInstance.charfeatures[i].size(); j++) {
				charfeatures[i].push_back(anInstance.charfeatures[i][j]);
			}
			for (int j = 0; j < anInstance.typefeatures[i].size(); j++) {
				typefeatures[i].push_back(anInstance.typefeatures[i][j]);
			}
		}

	}

	void assignLabel(const vector<string>& resulted_labels) {
		assert(resulted_labels.size() == words.size());
		labels.clear();
		for (int idx = 0; idx < resulted_labels.size(); idx++) {
			labels.push_back(resulted_labels[idx]);
		}
	}

	void assignAdditionLabel(const vector<string>& resulted_labels) {
		assert(resulted_labels.size() == words.size());
		additionlabels.clear();
		for (int idx = 0; idx < resulted_labels.size(); idx++) {
			additionlabels.push_back(resulted_labels[idx]);
		}
		useAddition = true;
	}

	void Evaluate(const vector<string>& resulted_labels, Metric& eval) const {
		for (int idx = 0; idx < labels.size(); idx++) {
			if (!validlabels(labels[idx]))
				continue;
			if (resulted_labels[idx].compare(labels[idx]) == 0)
				eval.correct_label_count++;
			eval.overall_label_count++;
		}
	}

  void SegEvaluate(const vector<string>& resulted_labels, Metric& eval) const {
    static int idx, idy, endpos;
    unordered_set<string> golds;
    // segmentation should be agree in both layers, usually, the first layer defines segmentation
    idx = 0;
    while (idx < labels.size()) {
      if (is_start_label(labels[idx])) {
        idy = idx;
        endpos = -1;
        while (idy < labels.size()) {
          if (!is_continue_label(labels[idy], labels[idx], idy - idx)) {
            endpos = idy - 1;
            break;
          }
          endpos = idy;
          idy++;
        }
        stringstream ss;
        ss << "[" << idx << "," << endpos << "]";
        golds.insert(cleanLabel(labels[idx]) + ss.str());
        idx = endpos;
      }
      idx++;
    }

    unordered_set<string> preds;
    idx = 0;
    while (idx < resulted_labels.size()) {
      if (is_start_label(resulted_labels[idx])) {
        stringstream ss;
        idy = idx;
        endpos = -1;
        while (idy < resulted_labels.size()) {
          if (!is_continue_label(resulted_labels[idy], resulted_labels[idx], idy - idx)) {
            endpos = idy - 1;
            break;
          }
          endpos = idy;
          idy++;
        }
        ss << "[" << idx << "," << endpos << "]";
        preds.insert(cleanLabel(resulted_labels[idx]) + ss.str());
        idx = endpos;
      }
      idx++;
    }

    unordered_set<string>::iterator iter;
    eval.overall_label_count += golds.size();
    eval.predicated_label_count += preds.size();
    for (iter = preds.begin(); iter != preds.end(); iter++) {
      if (golds.find(*iter) != golds.end()) {
        eval.correct_label_count++;
      }
    }

  }

public:
	vector<string> labels;
	vector<string> words;
	vector<vector<string> > typefeatures;
	vector<vector<string> > sparsefeatures;
	vector<vector<string> > charfeatures;

	vector<string> additionlabels;
	bool useAddition;

};

#endif

