#include "pch.h"

using namespace cv;

Mat colorBaseMat, colorBaseMat2, grayBaseMat, morpBaseMat, targetMat, bridgeMat;

void menuCheck();
bool ImageLoadProcess();

int main()
{
	if (!ImageLoadProcess())
	{
		return -1;
	}

	menuCheck();
	cvWaitKey();
	return 1;
}

bool ImageLoadProcess()
{
	colorBaseMat = imread("lena_color.jpg", CV_LOAD_IMAGE_COLOR);
	colorBaseMat2 = imread("dog.jpg", CV_LOAD_IMAGE_COLOR);
	morpBaseMat = imread("morphologyTest.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (colorBaseMat.empty() || colorBaseMat2.empty() || morpBaseMat.empty())
	{
		std::cout << "Image Load Failed!" << std::endl;
		return false;
	}

	grayBaseMat.create(Size(colorBaseMat.rows, colorBaseMat.cols), CV_8UC1);
	BasicImageProcess::ToGrayScale(colorBaseMat, grayBaseMat);

	return true;
}

void menuCheck()
{
	imshow("basic_Color", colorBaseMat);
	imshow("grayScale", grayBaseMat);
	cvWaitKey();

	enum MENU
	{
		EXIT, GRAYSCALE, INVERSE, YCBCR, CHECKHISTOGRAM, BINARY, DISSOLVE, LOWPASS, HIGHPASS, MEDIAN, MORPHOLOGY
	};

	int select;
	Histogram checkHist(256);
	double high_pass_filter[3][3] = { {-1, -1, -1},{-1, 8, -1},{-1, -1, -1} };
	double low_pass_filter[3][3] = { {1.0 / 9, 1.0 / 9, 1.0 / 9},{1.0 / 9, 1.0 / 9, 1.0 / 9},{1.0 / 9, 1.0 / 9, 1.0 / 9} };

	while (true)
	{
		std::cout << "메뉴를 선택하십시오\n (0 : 종료 | 1 : grayscale | 2 : inverse | 3 : ycbcr | 4 : checkHistogram | 5 : binary(Basic) |"
				  << " \n 6 : dissolve | 7 : lowpass | 8 : highpass | 9 : median | 10 : morphology) : ";
		std::cin >> select;
		
		switch (select)
		{
		case EXIT :
			std::cout << "프로그램을 종료합니다." << std::endl;
			return;

		case GRAYSCALE : 
			targetMat.create(Size(colorBaseMat.rows, colorBaseMat.cols), CV_8UC1);
			BasicImageProcess::ToGrayScale(colorBaseMat, targetMat);
			imshow("grayScale", targetMat);
			cvWaitKey();
			break;

		case INVERSE:
			targetMat.create(Size(colorBaseMat.rows, colorBaseMat.cols), CV_8UC3);
			BasicImageProcess::InverseImage(colorBaseMat, targetMat);
			imshow("Inverse", targetMat);
			cvWaitKey();
			break;

		case YCBCR:
			targetMat.create(Size(colorBaseMat.rows, colorBaseMat.cols), CV_8UC3);
			BasicImageProcess::ToYCrCbColor(colorBaseMat, targetMat);
			imshow("YCbCr", targetMat);
			cvWaitKey();
			break;

		case CHECKHISTOGRAM:
			checkHist.CheckHistogram(grayBaseMat);
			//checkHist.printHistogram();
			checkHist.getMaxCount();
			checkHist.drawHistogram();
			imshow("Histogram | of Lena(GrayScale)", checkHist.histogram_image);
			cvWaitKey();
			break;

		case BINARY:
			targetMat.create(Size(colorBaseMat.rows, colorBaseMat.cols), CV_8UC1);
			BasicImageProcess::ToBinary(grayBaseMat, targetMat, 128);
			imshow("binary", targetMat);
			cvWaitKey();
			break;
		
		case DISSOLVE:
			targetMat.create(Size(colorBaseMat.rows, colorBaseMat.cols), CV_8UC3);
			BasicImageProcess::DissolveImage(colorBaseMat, colorBaseMat2, targetMat, 0.5);
			imshow("dissolved", targetMat);
			cvWaitKey();
			break;

		case LOWPASS:
			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			FilterImageProcess::Calculate3x3Filter(grayBaseMat, targetMat, low_pass_filter);
			imshow("lowFilter", targetMat);
			cvWaitKey();
			break;

		case HIGHPASS:
			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			FilterImageProcess::Calculate3x3Filter(grayBaseMat, targetMat, high_pass_filter);
			imshow("highFilter", targetMat);
			cvWaitKey();
			break;

		case MEDIAN :
			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			FilterImageProcess::MedianFilter(grayBaseMat, targetMat);
			imshow("median", targetMat);
			cvWaitKey();
			break;

		case MORPHOLOGY :
			targetMat.create(Size(morpBaseMat.rows, morpBaseMat.cols), CV_8UC1);
			bridgeMat.create(Size(morpBaseMat.rows, morpBaseMat.cols), CV_8UC1);
			
			morpBaseMat.copyTo(targetMat);
			BasicImageProcess::MorphologyDilation(morpBaseMat, targetMat);
			imshow("Dilation", targetMat);

			morpBaseMat.copyTo(targetMat);
			BasicImageProcess::MorphologyErosion(morpBaseMat, targetMat);
			imshow("Erosion", targetMat);

			morpBaseMat.copyTo(targetMat);
			BasicImageProcess::MorphologyOpening(morpBaseMat, bridgeMat, targetMat);
			imshow("Opening", targetMat);

			morpBaseMat.copyTo(targetMat);
			BasicImageProcess::MorphologyClosing(morpBaseMat, bridgeMat, targetMat);
			imshow("Closing", targetMat);
			cvWaitKey();
			break;
		}
	}
}