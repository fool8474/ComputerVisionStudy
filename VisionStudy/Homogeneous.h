#pragma once
#include "pch.h"

class Homogeneous
{
private:
	const double PI;
	void MatrixMul(double homoMatrix[3][3]);
	void MatrixMul(double homoCoord[1][3], double outputCoord[1][3]);
	double HMatrix[3][3];

	bool initMat;

public:
	Homogeneous();
	void ForwardingMapping(cv::Mat baseMat, cv::Mat forwardMappedMat);
	void BackwardingMapping(cv::Mat baseMat, cv::Mat backwardMappedMat);
	void Move(const int y, const int x);
	void Rotate(const int degree);
	void InverseRotate(const int degree);
	void PrintHMatrix();
	double GetRadius(const int degree);

	void InitHomo();
};