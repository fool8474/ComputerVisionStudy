#include "ImageProcessing.h"
#include "Homogeneous.h"
#include <limits.h>

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

	void MoravecEdgeDetect(cv::Mat& baseMat, cv::Mat& moravecMat, std::vector<cv::Point>& edgeVec, int threshold)
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
	
				if (minRe >= threshold)
				{
					
					if (DoNonMaximumSuppression(baseMat, y, x))
					{
						if (y > 40 && y < baseMat.rows - 40 && x > 40 && x < baseMat.cols - 40)
						{
							drawEdgePoint(moravecMat, y, x);
							edgeVec.push_back(cv::Point(x, y));
						}
					}
					
				}
			}
		}

	}

	bool DoNonMaximumSuppression(cv::Mat& baseMat, int y, int x)
	{
		int neighbor_pt[4][4] = { {0,-1,0,1},{1,0,-1,0},{1,1,-1,-1},{1,-1,-1,1} };

		int cur_px = baseMat.at<uchar>(y, x);

		for (int y_pt = -1; y_pt < 2; y_pt++)
		{
			for (int x_pt = -1; x_pt < 2; x_pt++)
			{
				int mov_px = baseMat.at<uchar>(y + y_pt, x + x_pt);

				if (cur_px < mov_px)
					return false;
			}
		}
		
		return true;
	}

	void drawEdgePoint(cv::Mat edgeMat, int y, int x)
	{
		cv::circle(edgeMat, cv::Point(x, y), 2, cv::Scalar(0,255,0));
	}

	void calHogEdges(cv::Mat& baseMat, std::vector<cv::Point>& edges, std::vector<std::vector<Histogram>> &edge_histograms)
	{
		cv::Mat y_edge_mat, x_edge_mat;
		cv::Mat orient_mat, magnit_mat;

		const int CELLSIZE = 8;
		const int BLOCKSIZE = 16;
		const int STRIDE = 2;

		y_edge_mat.create(baseMat.size(), CV_8UC1);
		x_edge_mat.create(baseMat.size(), CV_8UC1);
		orient_mat.create(baseMat.size(), CV_8UC1);
		magnit_mat.create(baseMat.size(), CV_8UC1);
		orient_mat.setTo(cv::Scalar(1));
		magnit_mat.setTo(cv::Scalar(1));

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

			int cur_hist_pos = 0;

			for (int y_block_mv = -BLOCKSIZE; y_block_mv <= BLOCKSIZE; y_block_mv += BLOCKSIZE * STRIDE)
			{
				for (int x_block_mv = -BLOCKSIZE; x_block_mv <= BLOCKSIZE; x_block_mv += BLOCKSIZE * STRIDE)
				{
					
					Histogram hist_1(8);
					Histogram hist_2(8);
					Histogram hist_3(8);
					Histogram hist_4(8);

					edge_hist.push_back(hist_1);
					edge_hist.push_back(hist_2);
					edge_hist.push_back(hist_3);
					edge_hist.push_back(hist_4);

					double block_sum_for_norm = 0;

					for (int y_mv = -CELLSIZE; y_mv < CELLSIZE; y_mv++)
					{
						for (int x_mv = -CELLSIZE; x_mv < CELLSIZE; x_mv++)
						{
							int cur_magnit = (int)magnit_mat.at<uchar>(y_mv + y_block_mv + edges[edges_pt].y, x_mv + x_block_mv + edges[edges_pt].x);

							block_sum_for_norm += pow(cur_magnit,2);
						}
					}

					block_sum_for_norm = sqrt(block_sum_for_norm);

					for (int y_mv = -CELLSIZE; y_mv < CELLSIZE; y_mv++)
					{
						for (int x_mv = -CELLSIZE; x_mv < CELLSIZE; x_mv++)
						{
							int position = GetQuadrantForHOG(y_mv, x_mv);
							int cur_orient = (int)orient_mat.at<uchar>(y_mv + y_block_mv + edges[edges_pt].y, x_mv + x_block_mv + edges[edges_pt].x);
							int cur_magnit = (int)magnit_mat.at<uchar>(y_mv + y_block_mv + edges[edges_pt].y, x_mv + x_block_mv + edges[edges_pt].x);

							edge_hist[cur_hist_pos * 4 + position - 1].hist[cur_orient] += (cur_magnit / block_sum_for_norm);
						}
					}

					cur_hist_pos++;
				}
			}
			
			edge_histograms.push_back(edge_hist);
		}
		
		/*
		for (int i = 0; i < edge_histograms.size(); i++)
		{
			for (int j = 0; j < edge_histograms[i].size(); j++)
			{
				edge_histograms[i][j].printHistogram();
			}

			std::cout << "--------------" << std::endl;
		}
		*/
		
		
		cv::imshow("y_edge", y_edge_mat);
		cv::imshow("x_edge", x_edge_mat);
		cv::imshow("magnitude", magnit_mat);
		cv::imshow("orient", orient_mat);
		/*
		cv::waitKey();
		*/
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

	void GetEuclideanDistance(std::vector<std::vector<Histogram>> &edge_histograms1, std::vector<std::vector<Histogram>> &edge_histograms2, std::vector<cv::Point> &pair_point)
	{
		double min_of_min = std::numeric_limits<double>::max();

		std::vector<double> min_distances;
		std::vector<cv::Point> pair_temp_point, pair_temp_point_2;

		for (int cur_point_1 = 0; cur_point_1 < edge_histograms1.size(); cur_point_1++)
		{
			double min_distance = std::numeric_limits<double>::max();
			int min_position = -1;
			for (int cur_point_2 = 0; cur_point_2 < edge_histograms2.size(); cur_point_2++)
			{
				/*std::cout << cur_point_1 << " " << cur_point_2 << std::endl;*/

				double cur_distance = 0;

				std::vector<Histogram> hist_vec_1 = edge_histograms1.at(cur_point_1);
				std::vector<Histogram> hist_vec_2 = edge_histograms2.at(cur_point_2);

				for (int cur_hist = 0; cur_hist < hist_vec_1.size(); cur_hist++)
				{
					std::vector<double> hist_1 = hist_vec_1.at(cur_hist).hist;
					std::vector<double> hist_2 = hist_vec_2.at(cur_hist).hist;

					for (int hist_idx = 0; hist_idx < hist_1.size(); hist_idx++)
					{
						cur_distance += pow(hist_1[hist_idx] - hist_2[hist_idx], 2);
					}

					if (sqrt(cur_distance) > min_distance)
					{
						break;
					}
				}

				cur_distance = sqrt(cur_distance);

				if (cur_distance <= min_distance)
				{
					min_position = cur_point_2;
					min_distance = cur_distance;
				}
			}

			if (min_of_min >= min_distance)
			{
				min_of_min = min_distance;
			}

			min_distances.push_back(min_distance);
			pair_temp_point.push_back(cv::Point(cur_point_1, min_position));
		}

		for (int cur_point_2 = 0; cur_point_2 < edge_histograms2.size(); cur_point_2++)
		{
			double min_distance = std::numeric_limits<double>::max();
			int min_position = -1;
			for (int cur_point_1 = 0; cur_point_1 < edge_histograms1.size(); cur_point_1++)
			{
				/*std::cout << cur_point_1 << " " << cur_point_2 << std::endl;*/

				double cur_distance = 0;

				std::vector<Histogram> hist_vec_1 = edge_histograms1.at(cur_point_1);
				std::vector<Histogram> hist_vec_2 = edge_histograms2.at(cur_point_2);

				for (int cur_hist = 0; cur_hist < hist_vec_1.size(); cur_hist++)
				{
					std::vector<double> hist_1 = hist_vec_1.at(cur_hist).hist;
					std::vector<double> hist_2 = hist_vec_2.at(cur_hist).hist;

					for (int hist_idx = 0; hist_idx < hist_1.size(); hist_idx++)
					{
						cur_distance += pow(hist_1[hist_idx] - hist_2[hist_idx], 2);
					}

					if (sqrt(cur_distance) > min_distance)
					{
						break;
					}
				}

				cur_distance = sqrt(cur_distance);

				if (cur_distance <= min_distance)
				{
					min_position = cur_point_1;
					min_distance = cur_distance;
				}
			}

			if (min_of_min >= min_distance)
			{
				min_of_min = min_distance;
			}

			min_distances.push_back(min_distance);
			pair_temp_point_2.push_back(cv::Point(min_position, cur_point_2));
		}

		for (int i = 0; i < pair_temp_point.size(); i++)
		{
			for (int j = 0; j < pair_temp_point_2.size(); j++)
			{
				if (pair_temp_point.at(i).x == pair_temp_point_2.at(j).x)
				{
					if (pair_temp_point.at(i).y == pair_temp_point_2.at(j).y)
					{
						if (min_of_min * 6 > min_distances.at(i))
						{
							pair_point.push_back(pair_temp_point.at(i));
						}
					}
				}
			}
		}
	}

	void MakeImageForDrawPairPoint(cv::Mat &targetMat, cv::Mat &mat1, cv::Mat &mat2)
	{
		targetMat.create(cv::Size(2000, 2000), CV_8UC3);
		
		for (int y = 0; y < mat1.rows; y++)
		{
			for (int x = 0; x < mat1.cols; x++)
			{
				targetMat.at<cv::Vec3b>(y, x)[0] = mat1.at<cv::Vec3b>(y, x)[0];
				targetMat.at<cv::Vec3b>(y, x)[1] = mat1.at<cv::Vec3b>(y, x)[1];
				targetMat.at<cv::Vec3b>(y, x)[2] = mat1.at<cv::Vec3b>(y, x)[2];
			}
		}

		for (int y = 0; y < mat2.rows; y++)
		{
			for (int x = 0; x < mat2.cols; x++)
			{
				targetMat.at<cv::Vec3b>(y, x + mat1.cols)[0] = mat2.at<cv::Vec3b>(y, x)[0];
				targetMat.at<cv::Vec3b>(y, x + mat1.cols)[1] = mat2.at<cv::Vec3b>(y, x)[1];
				targetMat.at<cv::Vec3b>(y, x + mat1.cols)[2] = mat2.at<cv::Vec3b>(y, x)[2];
			}
		}
	}

	void DrawPairByPoints(cv::Mat &targetMat, cv::Size img1_size, std::vector<cv::Point> &pair_point, std::vector<cv::Point> &edges_1, std::vector<cv::Point> &edges_2)
	{
		std::srand((unsigned int)time(NULL));

		for (int cur_pair = 0; cur_pair < pair_point.size(); cur_pair++)
		{
			cv::Point pair1 = edges_1.at(pair_point.at(cur_pair).x);
			cv::Point pair2 = edges_2.at(pair_point.at(cur_pair).y);

			cv::line(targetMat, pair1, cv::Point(pair2.x + img1_size.width, pair2.y), cv::Scalar(std::rand() % 255, std::rand() % 255, std::rand() % 255), 3);
		}
	}

	void Ransac(std::vector<cv::Point> &pair, int num_of_data, std::vector<cv::Point> &edges_1, std::vector<cv::Point> &edges_2, cv::Mat &homogeneous_mat)
	{
		cv::Mat for_homogeneous;
		cv::Mat for_homogeneous_2;

		double prob = 0.05;

		int num_of_test = std::log(1 - prob) / std::log(1 - pow(prob, num_of_data));

		std::vector<cv::Point> best_inline_pair;
		int best_pair = -1;

		std::cout << "test Number : " << num_of_test << std::endl;
		for (int test_case = 0; test_case < num_of_test; test_case++)
		{
			std::vector<cv::Point> cur_inline;

			std::vector<int> select_array;
			for (int i = 0; i < num_of_data; i++)
			{
				select_array.push_back(std::rand() % pair.size());
			}

			for_homogeneous.create(cv::Size(6, 6), CV_64FC1);
			for_homogeneous.setTo(cv::Scalar(0));
			for_homogeneous_2.create(cv::Size(1, 6), CV_64FC1);
			for_homogeneous_2.setTo(cv::Scalar(0));

			for (int i = 0; i < num_of_data; i++)
			{
				MakeRansacArrays(select_array, pair, i, edges_1, edges_2, for_homogeneous, for_homogeneous_2);
			}

			cv::Mat t_s = for_homogeneous.inv() * for_homogeneous_2;

			homogeneous_mat.create(cv::Size(3, 3), CV_64FC1);
			homogeneous_mat.setTo(cv::Scalar(0));

			homogeneous_mat.at<double>(0, 0) = t_s.at<double>(0, 0);
			homogeneous_mat.at<double>(1, 0) = t_s.at<double>(1, 0);
			homogeneous_mat.at<double>(2, 0) = t_s.at<double>(2, 0);
			homogeneous_mat.at<double>(0, 1) = t_s.at<double>(3, 0);
			homogeneous_mat.at<double>(1, 1) = t_s.at<double>(4, 0);
			homogeneous_mat.at<double>(2, 1) = t_s.at<double>(5, 0);
			homogeneous_mat.at<double>(2, 2) = 1;

			double avg = 0;

			/*
			PrintMat(for_homogeneous.inv(), false);
			std::cout << "----------------" << std::endl;
			PrintMat(homogeneous_mat, false);
			*/

			for (int cur_pair = 0; cur_pair < pair.size(); cur_pair++)
			{
				cv::Mat a_mat, b_mat, b_res_mat;
				a_mat.create(cv::Size(3, 1), CV_64FC1);
				b_mat.create(cv::Size(3, 1), CV_64FC1);

				a_mat.at<double>(0, 0) = edges_1.at(pair.at(cur_pair).x).x;
				a_mat.at<double>(0, 1) = edges_1.at(pair.at(cur_pair).x).y;
				a_mat.at<double>(0, 2) = 1;

				b_res_mat = a_mat * homogeneous_mat;
				
				b_mat.at<double>(0, 0) = edges_2.at(pair.at(cur_pair).y).x;
				b_mat.at<double>(0, 1) = edges_2.at(pair.at(cur_pair).y).y;
				b_mat.at<double>(0, 2) = 1;
				
				double cur_distance =  sqrt(pow(b_res_mat.at<double>(0, 0) - b_mat.at<double>(0, 0), 2) + pow(b_res_mat.at<double>(0, 1) - b_mat.at<double>(0, 1), 2));
				avg += cur_distance;

				if (cur_distance < 30)
				{
					cur_inline.push_back(pair.at(cur_pair));
				}
			}

			if (cur_inline.size() > best_inline_pair.size())
			{
				best_inline_pair = cur_inline;
				best_pair = test_case;
			}

			std::cout << "best_pair : " << best_inline_pair.size() << std::endl;
		}

		GetBestLineHomogeneous(best_inline_pair, edges_1, edges_2, for_homogeneous, for_homogeneous_2, homogeneous_mat);
	}

	void GetBestLineHomogeneous(std::vector<cv::Point> pairs, std::vector<cv::Point> &edges_1, std::vector<cv::Point> &edges_2, cv::Mat &for_homogeneous, cv::Mat &for_homogeneous_2, cv::Mat &homogeneous_mat)
	{
		for (int i = 0; i < pairs.size(); i++)
		{
			for (int j = 0; j < 2; j++)
			{
				for_homogeneous.at<double>(0 + 3 * j, 0 + 3 * j) += pow(edges_1.at(pairs.at(i).x).x, 2);
				for_homogeneous.at<double>(0 + 3 * j, 1 + 3 * j) += edges_1.at(pairs.at(i).x).x * edges_1.at(pairs.at(i).x).y;
				for_homogeneous.at<double>(0 + 3 * j, 2 + 3 * j) += edges_1.at(pairs.at(i).x).x;
				for_homogeneous.at<double>(1 + 3 * j, 0 + 3 * j) += edges_1.at(pairs.at(i).x).x * edges_1.at(pairs.at(i).x).y;
				for_homogeneous.at<double>(1 + 3 * j, 2 + 3 * j) += edges_1.at(pairs.at(i).x).y;
				for_homogeneous.at<double>(2 + 3 * j, 0 + 3 * j) += edges_1.at(pairs.at(i).x).x;
				for_homogeneous.at<double>(2 + 3 * j, 1 + 3 * j) += edges_1.at(pairs.at(i).x).y;
				for_homogeneous.at<double>(2 + 3 * j, 2 + 3 * j) += 1;
			}

			for_homogeneous.at<double>(4, 4) += pow(edges_1.at(pairs.at(i).x).y, 2);

			for_homogeneous_2.at<double>(0, 0) += edges_1.at(pairs.at(i).x).x * edges_2.at(pairs.at(i).y).x;
			for_homogeneous_2.at<double>(1, 0) += edges_1.at(pairs.at(i).x).y * edges_2.at(pairs.at(i).y).x;
			for_homogeneous_2.at<double>(2, 0) += edges_2.at(pairs.at(i).y).x;
			for_homogeneous_2.at<double>(3, 0) += edges_1.at(pairs.at(i).x).x * edges_2.at(pairs.at(i).y).y;
			for_homogeneous_2.at<double>(4, 0) += edges_1.at(pairs.at(i).x).y * edges_2.at(pairs.at(i).y).y;
			for_homogeneous_2.at<double>(5, 0) += edges_2.at(pairs.at(i).y).y;
		}

		cv::Mat t_s = for_homogeneous.inv() * for_homogeneous_2;

		homogeneous_mat.create(cv::Size(3, 3), CV_64FC1);
		homogeneous_mat.setTo(cv::Scalar(0));

		homogeneous_mat.at<double>(0, 0) = t_s.at<double>(0, 0);
		homogeneous_mat.at<double>(1, 0) = t_s.at<double>(1, 0);
		homogeneous_mat.at<double>(2, 0) = t_s.at<double>(2, 0);
		homogeneous_mat.at<double>(0, 1) = t_s.at<double>(3, 0);
		homogeneous_mat.at<double>(1, 1) = t_s.at<double>(4, 0);
		homogeneous_mat.at<double>(2, 1) = t_s.at<double>(5, 0);
		homogeneous_mat.at<double>(2, 2) = 1;

		PrintMat(homogeneous_mat, false);
	}

	void PrintMat(cv::Mat printMat, bool isUchar)
	{
		for (int y = 0; y < printMat.rows; y++)
		{
			for (int x = 0; x < printMat.cols; x++)
			{
				if (isUchar)
				{
					std::cout << printMat.at<uchar>(y, x) << " ";
				}

				else
				{
					std::cout << printMat.at<double>(y, x) << " ";
				}
			}
			std::cout << std::endl;
		}
	}

	void MakeRansacArrays(std::vector<int> select_array, std::vector<cv::Point> &pair, int i, std::vector<cv::Point> &edges_1, std::vector<cv::Point> &edges_2, cv::Mat &for_homogeneous, cv::Mat &for_homogeneous_2)
	{
		for_homogeneous.at<double>(1, 1) += pow(edges_1.at(pair.at(select_array.at(i)).x).y, 2);

		for (int j = 0; j < 2; j++)
		{
			for_homogeneous.at<double>(0 + 3 * j, 0 + 3 * j) += pow(edges_1.at(pair.at(select_array.at(i)).x).x, 2);
			for_homogeneous.at<double>(0 + 3 * j, 1 + 3 * j) += edges_1.at(pair.at(select_array.at(i)).x).x * edges_1.at(pair.at(select_array.at(i)).x).y;
			for_homogeneous.at<double>(0 + 3 * j, 2 + 3 * j) += edges_1.at(pair.at(select_array.at(i)).x).x;
			for_homogeneous.at<double>(1 + 3 * j, 0 + 3 * j) += edges_1.at(pair.at(select_array.at(i)).x).x * edges_1.at(pair.at(select_array.at(i)).x).y;
			for_homogeneous.at<double>(1 + 3 * j, 2 + 3 * j) += edges_1.at(pair.at(select_array.at(i)).x).y;
			for_homogeneous.at<double>(2 + 3 * j, 0 + 3 * j) += edges_1.at(pair.at(select_array.at(i)).x).x;
			for_homogeneous.at<double>(2 + 3 * j, 1 + 3 * j) += edges_1.at(pair.at(select_array.at(i)).x).y;
			for_homogeneous.at<double>(2 + 3 * j, 2 + 3 * j) += 1;
		}

		for_homogeneous.at<double>(4, 4) += pow(edges_1.at(pair.at(select_array.at(i)).x).y, 2);

		for_homogeneous_2.at<double>(0, 0) += edges_1.at(pair.at(select_array.at(i)).x).x * edges_2.at(pair.at(select_array.at(i)).y).x;
		for_homogeneous_2.at<double>(1, 0) += edges_1.at(pair.at(select_array.at(i)).x).y * edges_2.at(pair.at(select_array.at(i)).y).x;
		for_homogeneous_2.at<double>(2, 0) += edges_2.at(pair.at(select_array.at(i)).y).x;
		for_homogeneous_2.at<double>(3, 0) += edges_1.at(pair.at(select_array.at(i)).x).x * edges_2.at(pair.at(select_array.at(i)).y).y;
		for_homogeneous_2.at<double>(4, 0) += edges_1.at(pair.at(select_array.at(i)).x).y * edges_2.at(pair.at(select_array.at(i)).y).y;
		for_homogeneous_2.at<double>(5, 0) += edges_2.at(pair.at(select_array.at(i)).y).y;
	}

	void MakePanoramaResultMap(cv::Mat &homogeneous_mat, cv::Mat &result_mat, cv::Mat base_mat1, cv::Mat base_mat2)
	{
		result_mat.create(cv::Size(base_mat1.cols + base_mat2.cols, base_mat1.rows + base_mat2.rows), CV_8UC3);
		result_mat.setTo(cv::Scalar(255, 255, 255));

		cv::Mat to_move_pt;

		int min_pt_x = 999999;
		int min_pt_y = 999999;

		for (int y = 0; y < base_mat1.rows; y++)
		{
			for (int x = 0; x < base_mat1.cols; x++)
			{
				to_move_pt.create(cv::Size(3, 1), CV_64FC1);
				to_move_pt.at<double>(0, 0) = x;
				to_move_pt.at<double>(0, 1) = y;
				to_move_pt.at<double>(0, 2) = 1;

				to_move_pt = to_move_pt * homogeneous_mat.inv();
				/*std::cout << "moved    : " << int(to_move_pt.at<double>(0, 0)) - min_pt_y << " " << int(to_move_pt.at<double>(0, 1) - min_pt_x) << " " << to_move_pt.at<double>(0, 2) << std::endl;
				std::cout << "original : " << x << " " << y << " " << 1 << std::endl;*/

				if (min_pt_x >= int(to_move_pt.at<double>(0, 0)))
					min_pt_x = int(to_move_pt.at<double>(0, 0));

				if (min_pt_y >= int(to_move_pt.at<double>(0, 1)))
					min_pt_y = int(to_move_pt.at<double>(0, 1));
			}
		}
		
		for (int y = 0; y < base_mat2.rows; y++)
		{
			for (int x = 0; x < base_mat2.cols; x++)
			{
				result_mat.at<cv::Vec3b>(y + min_pt_y, x + min_pt_x)[0] = base_mat2.at<cv::Vec3b>(y, x)[0];
				result_mat.at<cv::Vec3b>(y + min_pt_y, x + min_pt_x)[1] = base_mat2.at<cv::Vec3b>(y, x)[1];
				result_mat.at<cv::Vec3b>(y + min_pt_y, x + min_pt_x)[2] = base_mat2.at<cv::Vec3b>(y, x)[2];
			}
		}

		for (int y = 0; y < base_mat1.rows; y++)
		{
			for (int x = 0; x < base_mat1.cols; x++)
			{
				to_move_pt.create(cv::Size(3, 1), CV_64FC1);
				to_move_pt.at<double>(0, 0) = x;
				to_move_pt.at<double>(0, 1) = y;
				to_move_pt.at<double>(0, 2) = 1;

				to_move_pt = to_move_pt * homogeneous_mat.inv();

				result_mat.at<cv::Vec3b>(y,x)[0] = base_mat1.at<cv::Vec3b>(int(to_move_pt.at<double>(0, 1)) - min_pt_y, int(to_move_pt.at<double>(0, 0)) - min_pt_x)[0];
				result_mat.at<cv::Vec3b>(y,x)[1] = base_mat1.at<cv::Vec3b>(int(to_move_pt.at<double>(0, 1)) - min_pt_y, int(to_move_pt.at<double>(0, 0)) - min_pt_x)[1];
				result_mat.at<cv::Vec3b>(y,x)[2] = base_mat1.at<cv::Vec3b>(int(to_move_pt.at<double>(0, 1)) - min_pt_y, int(to_move_pt.at<double>(0, 0)) - min_pt_x)[2];

			}
		}

		cv::imshow("panorama", result_mat);
		cv::waitKey();
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