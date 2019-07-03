#pragma once
#include "opencv2/opencv.hpp"
#include "histogram.h"

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
	void MoravecEdgeDetect(cv::Mat& baseMat, cv::Mat& moravecMat, std::vector<cv::Point>& edgeVec, int threshold);
	bool DoNonMaximumSuppression(cv::Mat & baseMat, int y, int x);
	void drawEdgePoint(cv::Mat edgeMat, int y, int x);
	void calHogEdges(cv::Mat& baseMat, std::vector<cv::Point>& edges, std::vector<std::vector<Histogram>> &edge_histograms);
	int GetQuadrantForHOG(int y, int x);
	void GetEuclideanDistance(std::vector<std::vector<Histogram>>& edge_histograms1, std::vector<std::vector<Histogram>>& edge_histograms2, std::vector<cv::Point>& pair_point);
	void MakeImageForDrawPairPoint(cv::Mat & targetMat, cv::Mat & mat1, cv::Mat & mat2);
	void DrawPairByPoints(cv::Mat & targetMat, cv::Size img1_size, std::vector<cv::Point>& pair_point, std::vector<cv::Point>& edges_1, std::vector<cv::Point>& edges_2);
	void Ransac(std::vector<cv::Point>& pair, int num_of_data, std::vector<cv::Point>& edges_1, std::vector<cv::Point>& edges_2, cv::Mat & homogeneous_mat);
	void GetBestLineHomogeneous(std::vector<cv::Point> pairs, std::vector<cv::Point> &edges_1, std::vector<cv::Point> &edges_2, cv::Mat &for_homogeneous, cv::Mat &for_homogeneous_2, cv::Mat &homogeneous_mat);
	void PrintMat(cv::Mat printMat, bool isUchar);
	void MakeRansacArrays(std::vector<int> select_array, std::vector<cv::Point>& pair, int i, std::vector<cv::Point>& edges_1, std::vector<cv::Point>& edges_2, cv::Mat & for_homogeneous, cv::Mat & for_homogeneous_2);
	void MakePanoramaResultMap(cv::Mat & homogeneous_mat, cv::Mat & result_mat, cv::Mat base_mat1, cv::Mat base_mat2);
}

namespace FilterImageProcess
{
	void Calculate3x3Filter(cv::Mat baseMat, cv::Mat filteredMat, double filter[3][3]);
	void Calculate5x5Filter(cv::Mat baseMat, cv::Mat filteredMat, double filter[5][5]);
	void MedianFilter(cv::Mat baseMat, cv::Mat medianMat);
	void SortArray3x3(int sort_array[9]);
	void PyramidFilter(const int matCount, cv::Mat baseMat, std::vector<cv::Mat> *filteredMats, double filter[5][5]);
}