#include "ImageProcessing.h"

namespace BasicImageProcess
{
	void ToGrayScale(cv::Mat baseMat, cv::Mat grayMat)
	{
		for (int y = 0; y < baseMat.rows; y++)
		{
			for (int x = 0; x < baseMat.cols; x++)
			{
				grayMat.at<uchar>(y, x) = (baseMat.at<cv::Vec3b>(y, x)[0] + baseMat.at <cv::Vec3b >(y, x)[1] + baseMat.at<cv::Vec3b>(y, x)[2]) / 3;
			}
		}
	}

	void InverseImage(cv::Mat baseMat, cv::Mat inverseMat)
	{
		for (int y = 0; y < baseMat.rows; y++)
		{
			for (int x = 0; x < baseMat.cols; x++)
			{
				for (int rgbPoint = 0; rgbPoint < 3; rgbPoint++) 
				{
					inverseMat.at<cv::Vec3b>(y, x)[rgbPoint] = 255 - baseMat.at<cv::Vec3b>(y, x)[rgbPoint];
				}
			}
		}
	}

	void ToYCrCbColor(cv::Mat baseMat, cv::Mat ycbcrMat)
	{
		for (int y = 0; y < baseMat.rows; y++)
		{
			for (int x = 0; x < baseMat.cols; x++)
			{
				int r = baseMat.at<cv::Vec3b>(y, x)[0];
				int g = baseMat.at<cv::Vec3b>(y, x)[1];
				int b = baseMat.at<cv::Vec3b>(y, x)[2];
				ycbcrMat.at<cv::Vec3b>(y, x)[0] = (  0 + r *  0.299	+ g *  0.587 + b *  0.114);
				ycbcrMat.at<cv::Vec3b>(y, x)[1] = (128 + r * -0.169 + g * -0.331 + b *  0.500);
				ycbcrMat.at<cv::Vec3b>(y, x)[2] = (128 + r *  0.500 + g * -0.419 + b * -0.081);
			}
		}
	}

	void ToBinary(cv::Mat baseMat, cv::Mat binMat, int threshold)
	{
		for (int y = 0; y < baseMat.rows; y++)
		{
			for (int x = 0; x < baseMat.cols; x++)
			{
				if (baseMat.at<uchar>(y, x) < threshold)
				{
					binMat.at<uchar>(y, x) = 0;
				}

				else
				{
					binMat.at<uchar>(y, x) = 255;
				}
			}
		}
	}

	void DissolveImage(cv::Mat baseMat1, cv::Mat baseMat2, cv::Mat dissolveMat, double alpha)
	{
		for (int y = 0; y < baseMat1.rows; y++)
		{
			for (int x = 0; x < baseMat1.cols; x++)
			{
				for (int rgbPoint = 0; rgbPoint < 3; rgbPoint++)
				{
					dissolveMat.at<cv::Vec3b>(y, x)[rgbPoint] = alpha * baseMat1.at<cv::Vec3b>(y, x)[rgbPoint] + (1 - alpha) * baseMat2.at<cv::Vec3b>(y, x)[rgbPoint];
				}
			}
		}
	}

	void MorphologyErosion(cv::Mat baseMat, cv::Mat morpOutput)
	{
		int point[4][2] = { {-1,0},{1,0},{0,-1},{0,1}};

		for (int y = 1; y < baseMat.rows-1; y++)
		{
			for (int x = 1; x < baseMat.cols-1; x++)
			{
				if (baseMat.at<uchar>(y,x) < 128)
					for (int pt_cnt = 0; pt_cnt < 4; pt_cnt++)
					{
						morpOutput.at<uchar>(y + point[pt_cnt][0], x + point[pt_cnt][1]) = 0;
					}
			}
		}
	}

	void MorphologyDilation(cv::Mat baseMat, cv::Mat morpOutput)
	{
		int point[4][2] = { {-1,0},{1,0},{0,-1},{0,1}};

		for (int y = 1; y < baseMat.rows-1; y++)
		{
			for (int x = 1; x < baseMat.cols-1; x++)
			{
				if (baseMat.at<uchar>(y, x) > 128)
					for (int pt_cnt = 0; pt_cnt < 4; pt_cnt++)
					{
						morpOutput.at<uchar>(y + point[pt_cnt][0], x + point[pt_cnt][1]) = 255;
					}
			}
		}
	}

	void MorphologyOpening(cv::Mat baseMat, cv::Mat morpEro, cv::Mat morpOutput)
	{
		MorphologyErosion(baseMat, morpEro);
		morpEro.copyTo(morpOutput);
		MorphologyDilation(morpEro, morpOutput);
	}

	void MorphologyClosing(cv::Mat baseMat, cv::Mat morpDil, cv::Mat morpOutput)
	{
		MorphologyDilation(baseMat, morpDil);
		morpDil.copyTo(morpOutput);
		MorphologyErosion(morpDil, morpOutput);
	}
}

namespace FilterImageProcess
{
	void Calculate3x3Filter(cv::Mat baseMat, cv::Mat filteredMat, double filter[3][3])
	{
		filteredMat.setTo(cv::Scalar(255));

		for (int y = 1; y < baseMat.rows - 1; y++)
		{
			for (int x = 1; x < baseMat.cols - 1; x++)
			{
				double sum_of_filter_cal = 0;

				for (int fil_y = -1; fil_y < 2; fil_y++)
				{
					for (int fil_x = -1; fil_x < 2; fil_x++)
					{
						sum_of_filter_cal += (baseMat.at<uchar>(y + fil_y, x + fil_x) * filter[fil_y + 1][fil_x + 1]);
					}
				}
				sum_of_filter_cal = cv::saturate_cast<uchar>(sum_of_filter_cal);

				filteredMat.at<uchar>(y, x) = sum_of_filter_cal;
			}
		}
	}

	void MedianFilter(cv::Mat baseMat, cv::Mat medianMat)
	{
		for (int y = 1; y < baseMat.rows-1; y++)
		{
			for (int x = 1; x < baseMat.cols - 1; x++)
			{
				int sorted_array[9] = {0,};
				int count = 0;
				for (int pos_y = -1; pos_y < 2; pos_y++)
				{
					for (int pos_x = -1; pos_x < 2; pos_x++)
					{
						sorted_array[count++] = baseMat.at<uchar>(y + pos_y, x + pos_x);
					}
				}

				sortArray3x3(sorted_array);

				medianMat.at<uchar>(y,x) = sorted_array[4];
			}
		}
	}

	void sortArray3x3(int sort_array[9])
	{
		for (int i = 0; i < 8; i++)
		{
			for (int j = i; j < 9; j++)
			{
				if (sort_array[i] < sort_array[j])
				{
					int temp = sort_array[i];
					sort_array[i] = sort_array[j];
					sort_array[j] = temp;
				}
			}
		}
	}
}