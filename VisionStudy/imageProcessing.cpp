#include "ImageProcessing.h"

namespace BasicImageProcess
{
	void ToGrayScale(cv::Mat targetMat, cv::Mat grayMat)
	{
		for (int y = 0; y < targetMat.rows; y++)
		{
			for (int x = 0; x < targetMat.cols; x++)
			{
				grayMat.at<uchar>(y, x) = (targetMat.at<cv::Vec3b>(y, x)[0] + targetMat.at <cv::Vec3b >(y, x)[1] + targetMat.at<cv::Vec3b>(y, x)[2]) / 3;
			}
		}
	}

	void InverseImage(cv::Mat targetMat, cv::Mat inverseMat)
	{
		for (int y = 0; y < targetMat.rows; y++)
		{
			for (int x = 0; x < targetMat.cols; x++)
			{
				for (int rgbPoint = 0; rgbPoint < 3; rgbPoint++) 
				{
					inverseMat.at<cv::Vec3b>(y, x)[rgbPoint] = 255 - targetMat.at<cv::Vec3b>(y, x)[rgbPoint];
				}
			}
		}
	}

	void ToYCrCbColor(cv::Mat targetMat, cv::Mat ycbcrMat)
	{
		for (int y = 0; y < targetMat.rows; y++)
		{
			for (int x = 0; x < targetMat.cols; x++)
			{
				int r = targetMat.at<cv::Vec3b>(y, x)[0];
				int g = targetMat.at<cv::Vec3b>(y, x)[1];
				int b = targetMat.at<cv::Vec3b>(y, x)[2];
				ycbcrMat.at<cv::Vec3b>(y, x)[0] = (  0 + r *  0.299	+ g *  0.587 + b *  0.114);
				ycbcrMat.at<cv::Vec3b>(y, x)[1] = (128 + r * -0.169 + g * -0.331 + b *  0.500);
				ycbcrMat.at<cv::Vec3b>(y, x)[2] = (128 + r *  0.500 + g * -0.419 + b * -0.081);
			}
		}
	}
}