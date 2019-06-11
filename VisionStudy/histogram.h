#pragma once
#include "pch.h"

class Histogram 
{
public :
	std::vector<int> hist;

public :
	Histogram(int numOfHist);
	void CheckHistogram(cv::Mat targetMat);
	void printHistogram();
};