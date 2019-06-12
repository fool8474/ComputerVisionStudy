#pragma once
#include "pch.h"

class Histogram 
{
public :
	std::vector<int> hist;
	cv::Mat histogram_image;

private : 
	int maxCount;

public :
	Histogram(int numOfHist);
	void CheckHistogram(cv::Mat targetMat);
	void printHistogram();
	void drawHistogram();
	void getMaxCount();
};