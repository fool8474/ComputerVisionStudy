#include "opencv2/opencv.hpp"
#include "histogram.h"


Histogram::Histogram(int numOfHist)
{
	for (int i = 0; i < numOfHist; i++)
	{
		hist.push_back(0);
	}
}

void Histogram::CheckHistogram(cv::Mat targetMat)
{
	for (int y = 0; y < targetMat.rows; y++)
	{
		for (int x = 0; x < targetMat.cols; x++)
		{
			hist[targetMat.at<uchar>(y, x)]++;
		}
	}
}

void Histogram::printHistogram()
{
	for (int i = 0; i < hist.size(); i++)
	{
		std::cout << hist[i] << " ";
		if (i % 10 == 0 && i != 0)
			std::cout << std::endl;
	}

	std::cout << std::endl;
}
