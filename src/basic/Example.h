/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include "MyLib.h"

using namespace std;
struct Feature {
public:
	vector<string> words;
	vector<string> types;
	vector<string> chars;
	vector<string> segs;
	vector<string> linear_features;
public:
	Feature() {
	}

	//virtual ~Feature() {
	//
	//}

	void clear() {
		words.clear();
		chars.clear();
		types.clear();
		segs.clear();
		linear_features.clear();
	}
};

class Example {

public:
	vector<vector<dtype> > m_labels;
	vector<vector<vector<dtype> > > m_seglabels;
	vector<Feature> m_features;

public:
	Example()
	{

	}
	virtual ~Example()
	{

	}

	void clear()
	{
		m_features.clear();
		clearVec(m_labels);
		clearVec(m_seglabels);		
	}


};


#endif /* SRC_EXAMPLE_H_ */
