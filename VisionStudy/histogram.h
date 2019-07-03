#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"

class Histogram 
{
public :
	std::vector <double> hist;
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