#pragma once
#include "pch.h"

namespace BasicImageProcess
{
	void ToGrayScale(cv::Mat baseMat, cv::Mat grayMat);
	void InverseImage(cv::Mat baseMat, cv::Mat inverseMat);
	void ToYCrCbColor(cv::Mat baseMat, cv::Mat ycbcrMat);
	void ToBinary(cv::Mat baseMat, cv::Mat binMat, int threshold);
	void DissolveImage(cv::Mat baseMat1, cv::Mat baseMat2, cv::Mat dissolveMat, double alpha);
	void MorphologyErosion(cv::Mat baseMat, cv::Mat morpOutput);
	void MorphologyDilation(cv::Mat baseMat, cv::Mat morpOutput);
	void MorphologyClosing(cv::Mat baseMat, cv::Mat morpDil, cv::Mat morpOutput);
	void MorphologyOpening(cv::Mat baseMat, cv::Mat morpEro, cv::Mat morpOutput);
}

namespace FilterImageProcess
{
	void Calculate3x3Filter(cv::Mat baseMat, cv::Mat filteredMat, double filter[3][3]);
	void MedianFilter(cv::Mat baseMat, cv::Mat medianMat);
	void SortArray3x3(int sort_array[9]);
}