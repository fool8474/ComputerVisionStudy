#include "opencv2/opencv.hpp"

using namespace cv;

int main()
{
	Mat mat = imread("lena_color.jpg", CV_LOAD_IMAGE_COLOR);
	if (mat.empty())
	{
		std::cout << "�̸� Ʋ��" << std::endl;
		return -1;
	}

	Mat copyMat;

	mat.copyTo(copyMat);

	imshow("Test", mat);
	imshow("Copy", copyMat);

	cvWaitKey();
}