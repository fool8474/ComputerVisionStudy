#include "pch.h"

using namespace cv;

Mat colorBaseMat, grayBaseMat, targetMat;

void menuCheck();

int main()
{
	colorBaseMat = imread("lena_color.jpg", CV_LOAD_IMAGE_COLOR);
	if (colorBaseMat.empty())
	{
		std::cout << "�̸� Ʋ��" << std::endl;
		return -1;
	}

	grayBaseMat.create(Size(colorBaseMat.rows, colorBaseMat.cols), CV_8UC1);
	BasicImageProcess::ToGrayScale(colorBaseMat, grayBaseMat);
	
	menuCheck();
}

void menuCheck()
{
	imshow("basic_Color", colorBaseMat);
	imshow("grayScale", grayBaseMat);
	cvWaitKey();

	enum MENU
	{
		EXIT, GRAYSCALE, INVERSE, YCBCR, CHECKHISTOGRAM, DRAWHISTOGRAM
	};

	int select;
	Histogram checkHist(256);

	while (true)
	{
		std::cout << "�޴��� �����Ͻʽÿ� (0 : ���� | 1 : grayscale | 2 : inverse | 3 : ycbcr | 4 : checkHistogram | )";
		std::cin >> select;
		
		switch (select)
		{
		case EXIT :
			std::cout << "���α׷��� �����մϴ�." << std::endl;
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
			imshow("Histogram", checkHist.histogram_image);
			cvWaitKey();
			break;
		}
	}
}