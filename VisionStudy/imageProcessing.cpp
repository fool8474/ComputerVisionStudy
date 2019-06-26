
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

	void GetEdgeStrength(cv::Mat sobelX, cv::Mat sobelY, cv::Mat strengthMat)
	{
		int edge_strength;
		int current_px;

		for (int y = 0; y < sobelX.rows; y++)
		{
			for (int x = 0; x < sobelX.cols; x++)
			{
				edge_strength = (sqrt(pow(sobelX.at<uchar>(y, x), 2) + pow(sobelY.at<uchar>(y, x), 2)));
				current_px = cv::saturate_cast<uchar>(edge_strength);
				strengthMat.at<uchar>(y, x) = current_px;
			}
		}
	}

	void MoravecEdgeDetect(cv::Mat baseMat, cv::Mat moravecMat, std::vector<cv::Point>* edgeVec, int threshold)
	{

		using namespace std;

		int pt_dir[4][2] = { {-1,0}, {1,0}, {0,-1}, {0,1} };
		int pt_mora[9][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,0},{0,1},{1,-1},{1,0},{1,1} };
		
		int dir_catch_min[4] = {0,};

		for (int y = 2; y < baseMat.rows - 2; y++)
		{
			for (int x = 2; x < baseMat.cols - 2; x++)
			{
				bool check_edge = true;

				for(int i=0; i<4; i++)
				{ 
					dir_catch_min[i] = { 0 };
				}
				
				for (int cur_pt_dir = 0; cur_pt_dir < 4; cur_pt_dir++)
				{

					int cur_pt_result = 0;

					for (int cur_pt_mora = 0; cur_pt_mora < 9; cur_pt_mora++)
					{
						int ori_y_pos = baseMat.at<uchar>(y + pt_mora[cur_pt_mora][0], x + pt_mora[cur_pt_mora][1]);
						int mov_y_pos = baseMat.at<uchar>(y + pt_dir[cur_pt_dir][0] + pt_mora[cur_pt_mora][0], x + pt_dir[cur_pt_dir][1] + pt_mora[cur_pt_mora][1]);

						int cur_dir_result = pow(mov_y_pos - ori_y_pos, 2);

						cur_pt_result += cur_dir_result;
					}

					if (cur_pt_result <= threshold)
					{
						check_edge = false;
						break;
					}

					dir_catch_min[cur_pt_dir] = cur_pt_result;
				}

				int minRe = 99999999;
				int minPo = -1;

				for (int i = 0; i < 4; i++)
				{
					if (dir_catch_min[i] < minRe)
					{
						minRe = dir_catch_min[i];
						minPo = i;
					}
				}
	
				if (minRe >= 30000)
				{
					edgeVec->push_back(cv::Point(x,y));
					drawEdgePoint(moravecMat, y, x);
				}
			}
		}
	}

	void drawEdgePoint(cv::Mat edgeMat, int y, int x)
	{
		cv::circle(edgeMat, cv::Point(x, y), 2, cv::Scalar(0,255,0));
	}

	void calHogEdges(cv::Mat& baseMat, std::vector<cv::Point>& edges, std::vector<Histogram>& hists)
	{
		Histogram hist(10);
		hists.push_back(hist);
		for (int i = 0; i < 10; i++)
		{
			hist.hist[i] ++;
		}
		hist.printHistogram();
		return;
		cv::Mat y_edge_mat, x_edge_mat;
		cv::Mat orient_mat, magnit_mat;

		y_edge_mat.create(baseMat.size(), CV_8UC1);
		x_edge_mat.create(baseMat.size(), CV_8UC1);
		orient_mat.create(baseMat.size(), CV_8UC1);
		magnit_mat.create(baseMat.size(), CV_8UC1);
		
		for (int y = 1; y < baseMat.rows - 1; y++)
		{
			for (int x = 1; x < baseMat.cols - 1; x++)
			{
				int result_x = baseMat.at<uchar>(y, x + 1) - baseMat.at<uchar>(y, x - 1);
				int result_y = baseMat.at<uchar>(y + 1, x) - baseMat.at<uchar>(y - 1, x);

				float orientation = cv::fastAtan2(result_y, result_x); // atan2(result_y, (result_x + 0.000001)) / 3.141592 * 180;
				orientation = (int)(orientation / 45);
				orient_mat.at<uchar>(y, x) = orientation;

				int magnitude = (int)sqrt(pow(result_x, 2) + pow(result_y, 2));
				magnit_mat.at<uchar>(y, x) = magnitude;

				result_x = abs(result_x);
				result_y = abs(result_y);
				y_edge_mat.at<uchar>(y, x) = result_y;
				x_edge_mat.at<uchar>(y, x) = result_x;
				

			}
		}

		std::vector<std::vector<Histogram>> edges_all_hist;

		for (int edges_pt = 0; edges_pt < edges.size(); edges_pt++)
		{
			std::vector<Histogram> edge_hist;

			/* need to change 16 * 16 cell, 4 hist(8 grade) for edges */
			Histogram hist_1(8);
			Histogram hist_2(8);
			Histogram hist_3(8);
			Histogram hist_4(8);
			
			edge_hist.push_back(hist_1);
			edge_hist.push_back(hist_2);
			edge_hist.push_back(hist_3);
			edge_hist.push_back(hist_4);

			for (int y_mv = -8; y_mv < 8; y_mv++)
			{
				for (int x_mv = -8; x_mv < 8; x_mv++)
				{
					int position = GetQuadrantForHOG(y_mv, x_mv);
					int cur_orient = (int)orient_mat.at<uchar>(y_mv + edges[edges_pt].y, x_mv + edges[edges_pt].x);
					int cur_magnit = (int)magnit_mat.at<uchar>(y_mv + edges[edges_pt].y, x_mv + edges[edges_pt].x);
					edge_hist[position-1].hist[cur_orient] += cur_magnit;
				}
			}

			edges_all_hist.push_back(edge_hist);

		}
		
		cv::imshow("y_edge", y_edge_mat);
		cv::imshow("x_edge", x_edge_mat);
		cv::imshow("magnitude", magnit_mat);
		cv::imshow("orient", orient_mat);
		cv::waitKey();
	}

	int GetQuadrantForHOG(int y, int x)
	{
		if (y < 0 && x >= 0)
			return 1;

		if (y < 0 && x < 0)
			return 2;

		if (y >= 0 && x < 0)
			return 3;

		if (y >= 0 && x >= 0)
			return 4;
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

	void Calculate5x5Filter(cv::Mat baseMat, cv::Mat filteredMat, double filter[5][5])
	{
		filteredMat.setTo(cv::Scalar(255));

		for (int y = 2; y < baseMat.rows - 2; y++)
		{
			for (int x = 2; x < baseMat.cols - 2; x++)
			{
				double sum_of_filter_cal = 0;

				for (int fil_y = -2; fil_y < 3; fil_y++)
				{
					for (int fil_x = -2; fil_x < 3; fil_x++)
					{
						sum_of_filter_cal += (baseMat.at<uchar>(y + fil_y, x + fil_x) * filter[fil_y + 1][fil_x + 1]);
					}
				}
				sum_of_filter_cal = cv::saturate_cast<uchar>(sum_of_filter_cal);

				filteredMat.at<uchar>(y, x) = sum_of_filter_cal;
			}
		}
	}

	void PyramidFilter(const int matCount,  cv::Mat baseMat, std::vector<cv::Mat> *filteredMats, double filter[5][5])
	{
		
		filteredMats->push_back(baseMat);

		int newWidth = baseMat.cols;
		int newHeights = baseMat.rows;

		cv::Mat temp;
		baseMat.copyTo(temp);

		for (int i = 1; i < matCount; i++)
		{
			cv::Mat dst(cv::Size(newWidth / 2, newHeights / 2), CV_8UC1);
			dst.setTo(cv::Scalar(255));

			for (int y = 2; y < dst.rows-2; y++)
			{
				for (int x = 2; x < dst.cols-2; x++)
				{
					double sum_of_filter_cal = 0;

					for (int fil_y = -2; fil_y < 3; fil_y++)
					{
						for (int fil_x = -2; fil_x < 3; fil_x++)
						{
							sum_of_filter_cal += (filteredMats->at(i-1).at<uchar>(y * 2 + fil_y, x * 2 + fil_x) * filter[fil_y + 2][fil_x + 2]);
						}
					}

					sum_of_filter_cal = cv::saturate_cast<uchar>(sum_of_filter_cal);

					dst.at<uchar>(y, x) = sum_of_filter_cal;

				}
			}
			
			filteredMats->push_back(dst);

			newWidth /= 2;
			newHeights /= 2;
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

				SortArray3x3(sorted_array);

				medianMat.at<uchar>(y,x) = sorted_array[4];
			}
		}
	}

	void SortArray3x3(int sort_array[9])
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