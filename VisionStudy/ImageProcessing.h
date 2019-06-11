#pragma once
#include "pch.h"

namespace BasicImageProcess
{
	void ToGrayScale(cv::Mat targetMat, cv::Mat grayMat);
	void InverseImage(cv::Mat targetMat, cv::Mat inverseMat);
	void ToYCrCbColor(cv::Mat targetMat, cv::Mat ycbcrMat);
}

