#include "pch.h"

using namespace cv;

Mat colorBaseMat, colorBaseMat2, grayBaseMat, morpBaseMat, targetMat, targetMat2, bridgeMat;
Mat sobelXMat, sobelYMat, buckBaseMat, idolMat_1, idolMat_2, idolMat_result;
std::vector<Point> edges_1, edges_2;

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
	buckBaseMat = imread("bucks.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	idolMat_1 = imread("idol1.jpg", CV_LOAD_IMAGE_COLOR);
	idolMat_2 = imread("idol2.jpg", CV_LOAD_IMAGE_COLOR);

	if (colorBaseMat.empty() || colorBaseMat2.empty() || morpBaseMat.empty() || buckBaseMat.empty())
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
		EXIT = 0, GRAYSCALE, INVERSE, YCBCR, CHECKHISTOGRAM, BINARY, DISSOLVE, 
		LOWPASS = 7, HIGHPASS, MEDIAN, MORPHOLOGY, HOMOGENEOUSMOVE, HOMOGENEOUSROT,
		SOBELY = 13, SOBELX, PYRAMID, SOBELSTRENG, MORAVEC, HOG
	};

	int select;
	
	Histogram checkHist(256);
	Homogeneous homo;
	
	double high_pass_filter[3][3] = { {-1, -1, -1},{-1, 8, -1},{-1, -1, -1} };
	double low_pass_filter[3][3] = { {1.0 / 9, 1.0 / 9, 1.0 / 9},{1.0 / 9, 1.0 / 9, 1.0 / 9},{1.0 / 9, 1.0 / 9, 1.0 / 9} };
	double sobel_y_filter[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };
	double sobel_x_filter[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
	double pyramid_filter[5][5] = { {.0025, .0125, .0200, .0125, .0025},{.0125, .0625, .1000, .0625, .0125},{.0200, .1000, .1600, .1000, .0200},{.0125, .0625, .1000, .0625, .0125},{.0025, .0125, .0200, .0125, .0025} };
	
	std::vector<Mat> matVec;
	std::vector<Mat> * matVecPt = &matVec;

	while (true)
	{
		std::cout << "메뉴를 선택하십시오\n (0 : 종료 | 1 : grayscale | 2 : inverse | 3 : ycbcr | 4 : checkHistogram | 5 : binary(Basic) |"
				  << " \n 6 : dissolve | 7 : lowpass | 8 : highpass | 9 : median | 10 : morphology | 11 : homoMove | 12 : homoRot | "
				  << " \n 13 : sobelY | 14 : sobelX | 15 : pyramid | 16 : sobelStrength | 17 : MORAVEC | 18 : HOG) : ";
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

		case HOMOGENEOUSMOVE:	
			homo.InitHomo();
			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			homo.Move(50,50);
			homo.ForwardingMapping(grayBaseMat, targetMat);
			homo.PrintHMatrix();

			imshow("homoMove", targetMat);
			cvWaitKey();
			break;

		case HOMOGENEOUSROT:
			homo.InitHomo();
			homo.PrintHMatrix();

			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			homo.Move(-(grayBaseMat.rows / 2), -(grayBaseMat.cols / 2));
			homo.ForwardingMapping(grayBaseMat, targetMat);
			homo.PrintHMatrix();
			imshow("homo3Move", targetMat);
			cvWaitKey();

			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			homo.Rotate(70);
			homo.ForwardingMapping(grayBaseMat, targetMat);
			homo.PrintHMatrix();
			imshow("homo3MoveRot", targetMat);
			cvWaitKey();

			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			homo.Move(grayBaseMat.rows / 2, grayBaseMat.cols / 2);
			homo.ForwardingMapping(grayBaseMat, targetMat);
			homo.PrintHMatrix();
			imshow("homo3MoveRotMove", targetMat);
			cvWaitKey();

			break;

		case SOBELY :
			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			FilterImageProcess::Calculate3x3Filter(grayBaseMat, targetMat, sobel_y_filter);
			imshow("sobelYFilter", targetMat);
			cvWaitKey();
			break;

		case SOBELX :
			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			FilterImageProcess::Calculate3x3Filter(grayBaseMat, targetMat, sobel_x_filter);
			imshow("sobelXFilter", targetMat);
			cvWaitKey();
			break;

		case PYRAMID :
			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			FilterImageProcess::PyramidFilter(5, grayBaseMat, matVecPt, pyramid_filter);

			for (int i = 0; i < matVecPt->size(); i++)
			{
				std::string str = std::to_string(i) + "pyramid";
				cvNamedWindow(str.c_str(), CV_WINDOW_NORMAL);
				cv::imshow(str.c_str(), matVecPt->at(i));
			}

			cvWaitKey();

		case SOBELSTRENG :
			sobelXMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			FilterImageProcess::Calculate3x3Filter(grayBaseMat, sobelXMat, sobel_x_filter);
			
			sobelYMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);
			FilterImageProcess::Calculate3x3Filter(grayBaseMat, sobelYMat, sobel_y_filter);
			
			targetMat.create(Size(grayBaseMat.rows, grayBaseMat.cols), CV_8UC1);

			BasicImageProcess::GetEdgeStrength(sobelXMat, sobelYMat, targetMat);
			cv::imshow("edgeStrength", targetMat);

			cvWaitKey();
			break;

		case MORAVEC :
			
			cv::cvtColor(idolMat_1, targetMat, COLOR_BGR2GRAY);
			BasicImageProcess::MoravecEdgeDetect(targetMat, idolMat_1, &edges_1, 15000);
			cv::imshow("idol1", idolMat_1);
			cvWaitKey();

			cv::cvtColor(idolMat_2, targetMat, COLOR_BGR2GRAY);
			BasicImageProcess::MoravecEdgeDetect(targetMat, idolMat_2, &edges_2, 15000);
			cv::imshow("idol2", idolMat_2);
			cvWaitKey();
			break;

		case HOG :
			
			cv::cvtColor(idolMat_1, targetMat, COLOR_BGR2GRAY);
			BasicImageProcess::MoravecEdgeDetect(targetMat, idolMat_1, &edges_1, 15000);
			cv::imshow("idol1", idolMat_1);
			cvWaitKey();

			cv::cvtColor(idolMat_2, targetMat, COLOR_BGR2GRAY);
			BasicImageProcess::MoravecEdgeDetect(targetMat2, idolMat_2, &edges_2, 15000);
			cv::imshow("idol2", idolMat_2);
			cvWaitKey();

			std::vector<Histogram> hists;
			Histogram h1(10);
			hists.push_back(h1);
			BasicImageProcess::calHogEdges(targetMat, edges_1, hists);
			//BasicImageProcess::calHogEdges(targetMat2, edges_2);

			break;
		}
	}
}