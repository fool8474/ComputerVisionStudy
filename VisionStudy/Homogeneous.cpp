#include "Homogeneous.h"

void Homogeneous::InitHomo()
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			HMatrix[i][j] = 1;
		}
	}

	initMat = false;
}

void Homogeneous::MatrixMul(double homoMatrix[3][3])
{
	if (!initMat)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				HMatrix[i][j] = homoMatrix[i][j];
			}
		}

		initMat = true;
	}

	else
	{
		double mult[3][3] = { 0., };

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					mult[i][j] += HMatrix[i][k] * homoMatrix[k][i];
				}
			}
		}

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				HMatrix[i][j] = mult[i][j];
			}
		}
	}
}

void Homogeneous::MatrixMul(double homoCoord[1][3], double outputCoord[1][3])
{
	for (int i = 0; i < 1; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 3; ++k)
				outputCoord[i][j] += homoCoord[i][k] * HMatrix[k][j];
}

void Homogeneous::ForwardingMapping(cv::Mat baseMat, cv::Mat forwardMappedMat)
{
	forwardMappedMat.setTo(cv::Scalar(255));
	
	for (int y = 0; y < baseMat.rows; y++)
	{
		for (int x = 0; x < baseMat.cols; x++)
		{
			double homoCoord[1][3] = { 0., };
			double curPosition[1][3] = { y,x,1 };
			MatrixMul(curPosition, homoCoord);

			if (!(homoCoord[0][0] < 0) && !(homoCoord[0][0] >= baseMat.rows) && !(homoCoord[0][1] < 0) && !(homoCoord[0][1] >= baseMat.cols))
			{
				forwardMappedMat.at<uchar>(homoCoord[0][0], homoCoord[0][1]) = baseMat.at<uchar>(y, x);
			}
		}
	}
}

void Homogeneous::BackwardingMapping(cv::Mat baseMat, cv::Mat backwardMappedMat)
{

}

void Homogeneous::Move(const int y, const int x)
{
	double move_homo_matrix[3][3] = { { 1, 0, 0},{ 0, 1, 0},{ y, x, 1} };
	MatrixMul(move_homo_matrix);
}

void Homogeneous::Rotate(const int degree)
{
	double radian = GetRadius(degree);
	double rot_homo_matrix[3][3] = { { cos(radian), -sin(radian), 0},{ sin(radian), cos(radian), 0},{ 0, 0, 1} };
	MatrixMul(rot_homo_matrix);
}

void Homogeneous::InverseRotate(const int degree)
{

}

double Homogeneous::GetRadius(const int degree)
{
	return degree * (PI / 180);
}

void Homogeneous::PrintHMatrix() {
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << HMatrix[i][j] << " ";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;
}

Homogeneous::Homogeneous() : PI(acos(-1))
{
	initMat = false;
}