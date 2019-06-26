#pragma once
#include "histogram.h"
#include "pch.h"

class Histogram;

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
	void GetEdgeStrength(cv::Mat sobelX, cv::Mat sobelY, cv::Mat strengthMat);
	void MoravecEdgeDetect(cv::Mat baseMat, cv::Mat moravecMat, std::vector<cv::Point> *edgeVec, int threshold);
	void drawEdgePoint(cv::Mat edgeMat, int y, int x);
	void calHogEdges(cv::Mat& baseMat, std::vector<cv::Point>& edges, std::vector<Histogram>& hists);
	int GetQuadrantForHOG(int y, int x);
}

namespace FilterImageProcess
{
	void Calculate3x3Filter(cv::Mat baseMat, cv::Mat filteredMat, double filter[3][3]);
	void Calculate5x5Filter(cv::Mat baseMat, cv::Mat filteredMat, double filter[5][5]);
	void MedianFilter(cv::Mat baseMat, cv::Mat medianMat);
	void SortArray3x3(int sort_array[9]);
	void PyramidFilter(const int matCount, cv::Mat baseMat, std::vector<cv::Mat> *filteredMats, double filter[5][5]);
}