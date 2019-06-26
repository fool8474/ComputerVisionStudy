#include "histogram.h"

Histogram::Histogram(int numOfHist)
{
	for (int i = 0; i < numOfHist; i++)
	{
		hist.push_back(0);
	}
	
	histogram_image.create(cv::Size(hist.size()*3, hist.size()), CV_8UC3);
	histogram_image.setTo(cv::Scalar(0,0,0));
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

void Histogram::getMaxCount()
{
	maxCount = 0;

	for (int i = 0; i < hist.size(); i++)
	{
		if (maxCount < hist[i])
		{
			maxCount = hist[i];
		}
	}
}

void Histogram::drawHistogram()
{
	int lastPosition = 0;
	std::cout << histogram_image.rows << " " << histogram_image.cols << std::endl; // 256 , 768
	for (int i = 0; i < hist.size(); i++)
	{
		int curPosition = (int)(((double)hist[i] / maxCount) * histogram_image.rows);
		if (curPosition >= histogram_image.rows) curPosition = histogram_image.rows - 1;
		cv::Point pt1(i * 3, histogram_image.rows - curPosition);
		cv::Point pt2(i * 3, histogram_image.rows - lastPosition);
		lastPosition = curPosition;

		if(i != 0)	line(histogram_image, pt1, pt2, cv::Scalar(0, 255, 0), 3);
	}
}