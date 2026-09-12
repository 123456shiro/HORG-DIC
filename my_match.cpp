#include "my_match.h"

#include<opencv2\imgproc\types_c.h>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\features2d\features2d.hpp>	//特征提取
#include <algorithm>
#include<algorithm>
#include<vector>
#include<cmath>
#include <numeric>		
#include <random>//用于容器元素求和

using namespace cv;
using namespace std;

namespace opencorr
{
	myMatch::myMatch()
	{
	}
	myMatch::myMatch(double r)
	{
		dis_ratio = r;
	}
	Mat myMatch::LMSget(const Mat& match1_xy, const Mat& match2_xy, string model, float& rmse)//求仿射矩阵
	{
		if (match1_xy.rows != match2_xy.rows)
			CV_Error(CV_StsBadArg, "LMS模块输入特征点对个数不一致！");

		if (!(model == string("affine") || model == string("similarity") ||
			model == string("perspective") || model == string("projective")))
			CV_Error(CV_StsBadArg, "LMS模块图像变换类型输入错误！");

		const int N = match1_xy.rows;							//特征点个数
		Mat match2_xy_trans, match1_xy_trans;					//特征点坐标转置
		transpose(match1_xy, match1_xy_trans);					//矩阵转置(2,M)
		transpose(match2_xy, match2_xy_trans);

		Mat change = Mat::zeros(3, 3, CV_32FC1);				//变换矩阵

		//A*X=B,接下来部分仿射变换和透视变换一样,如果特征点个数是M，则A=[2*M,6]矩阵
		//A=[x1,y1,0,0,1,0;0,0,x1,y1,0,1;.....xn,yn,0,0,1,0;0,0,xn,yn,0,1]，应该是改版过的
		Mat A = Mat::zeros(2 * N, 6, CV_32FC1);

		for (int i = 0; i < N; ++i)
		{
			A.at<float>(2 * i, 0) = match2_xy.at<float>(i, 0);//x
			A.at<float>(2 * i, 1) = match2_xy.at<float>(i, 1);//y
			A.at<float>(2 * i, 4) = 1.f;

			A.at<float>(2 * i + 1, 2) = match2_xy.at<float>(i, 0);
			A.at<float>(2 * i + 1, 3) = match2_xy.at<float>(i, 1);
			A.at<float>(2 * i + 1, 5) = 1.f;
		}

		//如果特征点个数是M,那个B=[2*M,1]矩阵
		//B=[u1,v1,u2,v2,.....,un,vn]
		Mat B;

		B.create(2 * N, 1, CV_32FC1);
		for (int i = 0; i < N; ++i)
		{
			B.at<float>(2 * i, 0) = match1_xy.at<float>(i, 0);	  //x
			B.at<float>(2 * i + 1, 0) = match1_xy.at<float>(i, 1);//y
		}

		//如果是仿射变换
		if (model == string("affine"))
		{
			Vec6f values;
			solve(A, B, values, DECOMP_QR);
			change = (Mat_<float>(3, 3) << values(0), values(1), values(4),
				values(2), values(3), values(5),
				+0.0f, +0.0f, 1.0f);

			Mat temp_1 = change(Range(0, 2), Range(0, 2));		//尺度和旋转量
			Mat temp_2 = change(Range(0, 2), Range(2, 3));		//平移量

			Mat match2_xy_change = temp_1 * match2_xy_trans + repeat(temp_2, 1, N);
			Mat diff = match2_xy_change - match1_xy_trans;		//求差
			pow(diff, 2.f, diff);
			rmse = (float)sqrt(sum(diff)(0) * 1.0) / N;			//sum输出是各个通道的和, / N 初始实在括号里面
		}

		//如果是透视变换
		else if (model == string("perspective"))
		{
			/*透视变换模型
			[u'*w,v'*w, w]'=[u,v,w]' = [a1, a2, a5;
										a3, a4, a6;
										a7, a8, 1] * [x, y, 1]'
			[u',v']'=[x,y,0,0,1,0,-u'x, -u'y;
					 0, 0, x, y, 0, 1, -v'x,-v'y] * [a1, a2, a3, a4, a5, a6, a7, a8]'
			即，Y = A*X     */

			//构造 A 矩阵的后两列
			Mat A2;
			A2.create(2 * N, 2, CV_32FC1);
			for (int i = 0; i < N; ++i)
			{
				A2.at<float>(2 * i, 0) = match1_xy.at<float>(i, 0) * match2_xy.at<float>(i, 0) * (-1.f);
				A2.at<float>(2 * i, 1) = match1_xy.at<float>(i, 0) * match2_xy.at<float>(i, 1) * (-1.f);

				A2.at<float>(2 * i + 1, 0) = match1_xy.at<float>(i, 1) * match2_xy.at<float>(i, 0) * (-1.f);
				A2.at<float>(2 * i + 1, 1) = match1_xy.at<float>(i, 1) * match2_xy.at<float>(i, 1) * (-1.f);
			}

			Mat A1;											//完成的 A 矩阵，(8,8)
			A1.create(2 * N, 8, CV_32FC1);
			A.copyTo(A1(Range::all(), Range(0, 6)));
			A2.copyTo(A1(Range::all(), Range(6, 8)));

			Mat values;										//values中存放的是待求的8个参数
			solve(A1, B, values, DECOMP_QR);				//(2N,8)*(8,1)=(2N,1)好像本身就有点超定矩阵的意思
			change.at<float>(0, 0) = values.at<float>(0);
			change.at<float>(0, 1) = values.at<float>(1);
			change.at<float>(0, 2) = values.at<float>(4);
			change.at<float>(1, 0) = values.at<float>(2);
			change.at<float>(1, 1) = values.at<float>(3);
			change.at<float>(1, 2) = values.at<float>(5);
			change.at<float>(2, 0) = values.at<float>(6);
			change.at<float>(2, 1) = values.at<float>(7);
			change.at<float>(2, 2) = 1.f;

			Mat temp1 = Mat::ones(1, N, CV_32FC1);
			Mat temp2;
			temp2.create(3, N, CV_32FC1);

			match2_xy_trans.copyTo(temp2(Range(0, 2), Range::all()));
			temp1.copyTo(temp2(Range(2, 3), Range::all()));

			Mat match2_xy_change = change * temp2;		   //待配准图像中的特征点在参考图中的映射结果
			Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());
			//float* temp_ptr = match2_xy_change.ptr<float>(2);

			float* temp_ptr = match2_xy_change.ptr<float>(2);

			for (int i = 0; i < N; ++i)
			{
				float div_temp = temp_ptr[i];
				match2_xy_change_12.at<float>(0, i) = match2_xy_change_12.at<float>(0, i) / div_temp;
				match2_xy_change_12.at<float>(1, i) = match2_xy_change_12.at<float>(1, i) / div_temp;
			}

			Mat diff = match2_xy_change_12 - match1_xy_trans;
			pow(diff, 2, diff);

			rmse = (float)sqrt(sum(diff)(0) * 1.0) / N;	  //sum输出是各个通道的和，rmse为输入点的均方误差
		}

		//如果是相似变换
		else if (model == string("similarity"))
		{
			/*[x, y, 1, 0;
			  y, -x, 0, 1] * [a, b, c, d]'=[u,v]*/

			Mat A3;
			A3.create(2 * N, 4, CV_32FC1);
			for (int i = 0; i < N; ++i)
			{
				A3.at<float>(2 * i, 0) = match2_xy.at<float>(i, 0);
				A3.at<float>(2 * i, 1) = match2_xy.at<float>(i, 1);
				A3.at<float>(2 * i, 2) = 1.f;
				A3.at<float>(2 * i, 3) = 0.f;

				A3.at<float>(2 * i + 1, 0) = match2_xy.at<float>(i, 1);
				A3.at<float>(2 * i + 1, 1) = match2_xy.at<float>(i, 0) * (-1.f);
				A3.at<float>(2 * i + 1, 2) = 0.f;
				A3.at<float>(2 * i + 1, 3) = 1.f;
			}

			Vec4f values;
			solve(A3, B, values, DECOMP_QR);
			change = (Mat_<float>(3, 3) << values(0), values(1), values(2),
				values(1) * (-1.0f), values(0), values(3),
				+0.f, +0.f, 1.f);

			Mat temp_1 = change(Range(0, 2), Range(0, 2));//尺度和旋转量
			Mat temp_2 = change(Range(0, 2), Range(2, 3));//平移量

			Mat match2_xy_change = temp_1 * match2_xy_trans + repeat(temp_2, 1, N);
			Mat diff = match2_xy_change - match1_xy_trans;//求差
			pow(diff, 2, diff);
			rmse = (float)sqrt(sum(diff)(0) * 1.0) / N;//sum输出是各个通道的和
		}

		return change;
	}

	Mat myMatch::improve_LMS(const Mat& match1_xy, const Mat& match2_xy, string model, float& rmse)
	{
		if (match1_xy.rows != match2_xy.rows)
			CV_Error(CV_StsBadArg, "LMS模块输入特征点对个数不一致！");

		if (!(model == string("affine") || model == string("similarity") ||
			model == string("perspective")))
			CV_Error(CV_StsBadArg, "LMS模块图像变换类型输入错误！");

		const int N = match1_xy.rows;							//特征点个数
		Mat match2_xy_trans, match1_xy_trans;					//特征点坐标转置
		transpose(match1_xy, match1_xy_trans);					//矩阵转置(2,M)
		transpose(match2_xy, match2_xy_trans);
		Mat change = Mat::zeros(3, 3, CV_32FC1);				//变换矩阵

		//A*X=B,接下来部分仿射变换和透视变换一样,如果特征点个数是M，则A=[2*M,6]矩阵
		//A=[x1,y1,0,0,1,0;0,0,x1,y1,0,1;.....xn,yn,0,0,1,0;0,0,xn,yn,0,1]，应该是改版过的
		Mat A = Mat::zeros(2 * N, 6, CV_32FC1);

		for (int i = 0; i < N; ++i)
		{
			A.at<float>(2 * i, 0) = match2_xy.at<float>(i, 0);//x
			A.at<float>(2 * i, 1) = match2_xy.at<float>(i, 1);//y
			A.at<float>(2 * i, 4) = 1.f;

			A.at<float>(2 * i + 1, 2) = match2_xy.at<float>(i, 0);
			A.at<float>(2 * i + 1, 3) = match2_xy.at<float>(i, 1);
			A.at<float>(2 * i + 1, 5) = 1.f;
		}

		//如果特征点个数是M,那个B=[2*M,1]矩阵
		//B=[u1,v1,u2,v2,.....,un,vn]
		Mat B;

		B.create(2 * N, 1, CV_32FC1);
		for (int i = 0; i < N; ++i)
		{
			B.at<float>(2 * i, 0) = match1_xy.at<float>(i, 0);	  //x
			B.at<float>(2 * i + 1, 0) = match1_xy.at<float>(i, 1);//y
		}

		//如果是仿射变换
		if (model == string("affine"))
		{
			Vec6f values;
			solve(A, B, values, DECOMP_QR);
			change = (Mat_<float>(3, 3) << values(0), values(1), values(4),
				values(2), values(3), values(5),
				+0.0f, +0.0f, 1.0f);

			Mat temp_1 = change(Range(0, 2), Range(0, 2));//尺度和旋转量
			Mat temp_2 = change(Range(0, 2), Range(2, 3));//平移量

			Mat match2_xy_change = temp_1 * match2_xy_trans + repeat(temp_2, 1, N);
			Mat diff = match2_xy_change - match1_xy_trans;//求差
			pow(diff, 2.f, diff);
			rmse = (float)sqrt(sum(diff)(0) * 1.0 / N);//sum输出是各个通道的和
		}
		//如果是透视变换
		else if (model == string("perspective"))
		{
			/*透视变换模型
			[u'*w,v'*w, w]'=[u,v,w]' = [a1, a2, a5;
										a3, a4, a6;
										a7, a8, 1] * [x, y, 1]'
			[u',v']'=[x,y,0,0,1,0,-u'x, -u'y;
					 0, 0, x, y, 0, 1, -v'x,-v'y] * [a1, a2, a3, a4, a5, a6, a7, a8]'
			即，Y = A*X     */

			//构造 A 矩阵的后两列
			Mat A2;
			A2.create(2 * N, 2, CV_32FC1);
			for (int i = 0; i < N; ++i)
			{
				A2.at<float>(2 * i, 0) = match1_xy.at<float>(i, 0) * match2_xy.at<float>(i, 0) * (-1.f);
				A2.at<float>(2 * i, 1) = match1_xy.at<float>(i, 0) * match2_xy.at<float>(i, 1) * (-1.f);

				A2.at<float>(2 * i + 1, 0) = match1_xy.at<float>(i, 1) * match2_xy.at<float>(i, 0) * (-1.f);
				A2.at<float>(2 * i + 1, 1) = match1_xy.at<float>(i, 1) * match2_xy.at<float>(i, 1) * (-1.f);
			}

			Mat A1;											 //完整的 A 矩阵，(8,8)
			A1.create(2 * N, 8, CV_32FC1);
			A.copyTo(A1(Range::all(), Range(0, 6)));
			A2.copyTo(A1(Range::all(), Range(6, 8)));

			Mat AA1, balance;
			Mat evects, evals;
			transpose(A1, AA1);								 //求矩阵 A1 的转置
			balance = AA1 * A1;

			double a[8][8];
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					a[i][j] = balance.at<float>(i, j);
				}
			}
			//构造输入矩阵
			CvMat SrcMatrix = cvMat(8, 8, CV_32FC1, a);

			double b[8][8] =
			{
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
			};
			// 构造输出特征向量矩阵,特征向量按行存储,且与特征值相对应
			CvMat ProVector = cvMat(8, 8, CV_32FC1, b);

			// 构造输出特征值矩阵,特征值按降序配列
			double c[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			CvMat ProValue = cvMat(8, 1, CV_32FC1, c);

			//求特征向量，最后一行特征向量即对应的 H 矩阵的参数
			cvEigenVV(&SrcMatrix, &ProVector, &ProValue, 1.0e-6F);

			//输出特征向量矩阵
				//for (int i = 0; i < 2; i++)
				//{
				//	for (int j = 0; j < 2; j++)
				//		printf("%f\t", cvmGet(&ProVector, i, j));
				//	printf("\n");
				//}

				//把计算得到的最小特征值对应的特征变量赋给 H 矩阵
				change.at<float>(0, 0) = cvmGet(&ProVector, 7, 0);
			change.at<float>(0, 1) = cvmGet(&ProVector, 7, 1);
			change.at<float>(0, 2) = cvmGet(&ProVector, 7, 2);
			change.at<float>(1, 0) = cvmGet(&ProVector, 7, 3);
			change.at<float>(1, 1) = cvmGet(&ProVector, 7, 4);
			change.at<float>(1, 2) = cvmGet(&ProVector, 7, 5);
			change.at<float>(2, 0) = cvmGet(&ProVector, 7, 6);
			change.at<float>(2, 1) = cvmGet(&ProVector, 7, 7);
			change.at<float>(2, 2) = 1.f;

			Mat temp1 = Mat::ones(1, N, CV_32FC1);
			Mat temp2;										//存放处理后的特征点(x,y,1)T
			temp2.create(3, N, CV_32FC1);

			match2_xy_trans.copyTo(temp2(Range(0, 2), Range::all()));
			temp1.copyTo(temp2(Range(2, 3), Range::all()));

			Mat match2_xy_change = change * temp2;		   //待配准图像中的特征点在参考图中的映射结果
			Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());
			//float* temp_ptr = match2_xy_change.ptr<float>(2);

			float* temp_ptr = match2_xy_change.ptr<float>(2);

			for (int i = 0; i < N; ++i)
			{
				float div_temp = temp_ptr[i];
				match2_xy_change_12.at<float>(0, i) = match2_xy_change_12.at<float>(0, i) / div_temp;
				match2_xy_change_12.at<float>(1, i) = match2_xy_change_12.at<float>(1, i) / div_temp;
			}

			Mat diff = match2_xy_change_12 - match1_xy_trans;
			pow(diff, 2, diff);

			rmse = (float)sqrt(sum(diff)(0) * 1.0 / N);	  //sum输出是各个通道的和，rmse为输入点的均方误差
		}
		//如果是相似变换
		else if (model == string("similarity"))
		{
			/*[x, y, 1, 0;
			  y, -x, 0, 1] * [a, b, c, d]'=[u,v]*/

			Mat A3;
			A3.create(2 * N, 4, CV_32FC1);
			for (int i = 0; i < N; ++i)
			{
				A3.at<float>(2 * i, 0) = match2_xy.at<float>(i, 0);
				A3.at<float>(2 * i, 1) = match2_xy.at<float>(i, 1);
				A3.at<float>(2 * i, 2) = 1.f;
				A3.at<float>(2 * i, 3) = 0.f;

				A3.at<float>(2 * i + 1, 0) = match2_xy.at<float>(i, 1);
				A3.at<float>(2 * i + 1, 1) = match2_xy.at<float>(i, 0) * (-1.f);
				A3.at<float>(2 * i + 1, 2) = 0.f;
				A3.at<float>(2 * i + 1, 3) = 1.f;
			}

			Vec4f values;
			solve(A3, B, values, DECOMP_QR);
			change = (Mat_<float>(3, 3) << values(0), values(1), values(2),
				values(1) * (-1.0f), values(0), values(3),
				+0.f, +0.f, 1.f);

			Mat temp_1 = change(Range(0, 2), Range(0, 2));//尺度和旋转量
			Mat temp_2 = change(Range(0, 2), Range(2, 3));//平移量

			Mat match2_xy_change = temp_1 * match2_xy_trans + repeat(temp_2, 1, N);
			Mat diff = match2_xy_change - match1_xy_trans;//求差
			pow(diff, 2, diff);
			rmse = (float)sqrt(sum(diff)(0) * 1.0) / N;//sum输出是各个通道的和
		}

		return change;
	}

	Mat myMatch::ransac(const vector<Point2f>& points_1, const vector<Point2f>& points_2, string model, float threshold, vector<bool>& inliers, float& rmse)
	{
		if (points_1.size() != points_2.size())
			CV_Error(CV_StsBadArg, "ransac模块输入特征点数量不一致！");

		if (!(model == string("affine") || model == string("similarity") ||
			model == string("perspective") || model == string("projective")))
			CV_Error(CV_StsBadArg, "ransac模块图像变换类型输入错误！");

		const size_t N = points_1.size();					//特征点对数，size_t 类型常用来保存一个数据的大小，通常为整形，可移植性强
		int n;												//相当于不同模型对应的标签
		size_t max_iteration, iterations;

		//确定最大迭代次数，就目前来看此法过于简单粗暴，可以使用自适应迭代次数法
		if (model == string("similarity"))
		{
			n = 2;
			max_iteration = N * (N - 1) / 2;
		}
		else if (model == string("affine"))
		{
			n = 3;
			max_iteration = N * (N - 1) * (N - 2) / (2 * 3);
		}
		else if (model == string("perspective"))
		{
			n = 4;
			max_iteration = N * (N - 1) * (N - 2) / (2 * 3);
		}

		if (max_iteration > 800)
			iterations = 800;
		else
			iterations = max_iteration;

		//取出保存在points_1和points_2中的点坐标，保存在Mat矩阵中，方便处理
		Mat arr_1, arr_2;									//arr_1,和arr_2是一个[3 x N]的矩阵，每一列表示一个点坐标,第三行全是1
		arr_1.create(3, N, CV_32FC1);
		arr_2.create(3, N, CV_32FC1);

		//获取矩阵每一行的首地址
		float* p10 = arr_1.ptr<float>(0), * p11 = arr_1.ptr<float>(1), * p12 = arr_1.ptr<float>(2);
		float* p20 = arr_2.ptr<float>(0), * p21 = arr_2.ptr<float>(1), * p22 = arr_2.ptr<float>(2);

		//把特征点放到矩阵中
		for (size_t i = 0; i < N; ++i)
		{
			p10[i] = points_1[i].x;
			p11[i] = points_1[i].y;
			p12[i] = 1.f;

			p20[i] = points_2[i].x;
			p21[i] = points_2[i].y;
			p22[i] = 1.f;
		}

		Mat rand_mat;								//特征点索引
		rand_mat.create(1, n, CV_32SC1);

		int* p = rand_mat.ptr<int>(0);

		Mat sub_arr1, sub_arr2;						//存放随机挑选的特征点
		sub_arr1.create(n, 2, CV_32FC1);
		sub_arr2.create(n, 2, CV_32FC1);

		Mat T;										//待配准图像到参考图像的变换矩阵
		int most_consensus_num = 0;					//当前最优一致集个数初始化为0

		vector<bool> right;
		right.resize(N);
		inliers.resize(N);

		for (size_t i = 0; i < iterations; ++i)		//迭代次数
		{
			//随机选择n个不同的点对，不同的模型每次随机选择的个数不同
			while (1)
			{
				randu(rand_mat, 0, N - 1);			//随机生成n个范围在[0,N-1]之间的数，作为获取特征点的索引

				//保证这n个点坐标不相同
				if (n == 2 && p[0] != p[1] &&
					(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
					(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]))
					break;

				if (n == 3 && p[0] != p[1] && p[0] != p[2] && p[1] != p[2] &&
					(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
					(p10[p[0]] != p10[p[2]] || p11[p[0]] != p11[p[2]]) &&
					(p10[p[1]] != p10[p[2]] || p11[p[1]] != p11[p[2]]) &&
					(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]) &&
					(p20[p[0]] != p20[p[2]] || p21[p[0]] != p21[p[2]]) &&
					(p20[p[1]] != p20[p[2]] || p21[p[1]] != p21[p[2]]))
					break;

				if (n == 4 && p[0] != p[1] && p[0] != p[2] && p[0] != p[3] &&
					p[1] != p[2] && p[1] != p[3] && p[2] != p[3] &&
					(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
					(p10[p[0]] != p10[p[2]] || p11[p[0]] != p11[p[2]]) &&
					(p10[p[0]] != p10[p[3]] || p11[p[0]] != p11[p[3]]) &&
					(p10[p[1]] != p10[p[2]] || p11[p[1]] != p11[p[2]]) &&
					(p10[p[1]] != p10[p[3]] || p11[p[1]] != p11[p[3]]) &&
					(p10[p[2]] != p10[p[3]] || p11[p[2]] != p11[p[3]]) &&
					(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]) &&
					(p20[p[0]] != p20[p[2]] || p21[p[0]] != p21[p[2]]) &&
					(p20[p[0]] != p20[p[3]] || p21[p[0]] != p21[p[3]]) &&
					(p20[p[1]] != p20[p[2]] || p21[p[1]] != p21[p[2]]) &&
					(p20[p[1]] != p20[p[3]] || p21[p[1]] != p21[p[3]]) &&
					(p20[p[2]] != p20[p[3]] || p21[p[2]] != p21[p[3]]))
					break;
			}

			//提取出n个点对
			for (int i = 0; i < n; ++i)
			{
				sub_arr1.at<float>(i, 0) = p10[p[i]];
				sub_arr1.at<float>(i, 1) = p11[p[i]];

				sub_arr2.at<float>(i, 0) = p20[p[i]];
				sub_arr2.at<float>(i, 1) = p21[p[i]];
			}

			//根据这n个点对，计算初始变换矩阵 T
			T = LMSget(sub_arr1, sub_arr2, model, rmse);

			int consensus_num = 0;					//当前一致集(内点)个数

			if (model == string("perspective"))
			{
				//变换矩阵计算待配准图像特征点在参考图像中的映射点
				Mat match2_xy_change = T * arr_2;	//arr_2 中存放的是待配准图像的特征点 (3,N),

				//match2_xy_change(Range(0, 2), Range::all())意思是提取 match2_xy_change 的 0、1 行，所有的列
				Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());

				//获取 match2_xy_change 第二行首地址
				float* temp_ptr = match2_xy_change.ptr<float>(2);

				for (size_t i = 0; i < N; ++i)
				{
					float div_temp = temp_ptr[i];	//match2_xy_change 第二行第 i 列值，除以 div_temp ，是为了保证第三行为 1，和原始坐标相对应
					match2_xy_change_12.at<float>(0, i) = match2_xy_change_12.at<float>(0, i) / div_temp;
					match2_xy_change_12.at<float>(1, i) = match2_xy_change_12.at<float>(1, i) / div_temp;
				}

				//计算待配准图像特征点在参考图像中的映射点与参考图像中对应点的距离
				Mat diff = match2_xy_change_12 - arr_1(Range(0, 2), Range::all());

				pow(diff, 2, diff);

				//第一行和第二行求和，即两点间距离的平方
				Mat add = diff(Range(0, 1), Range::all()) + diff(Range(1, 2), Range::all());

				float* p_add = add.ptr<float>(0);

				//遍历所有距离，如果小于阈值，则认为是局内点
				for (size_t i = 0; i < N; ++i)
				{
					if (p_add[i] < threshold)				//初始 p_add[i]		
					{
						right[i] = true;
						++consensus_num;
					}
					else
						right[i] = false;
				}
			}

			else if (model == string("affine") || model == string("similarity"))
			{
				Mat match2_xy_change = T * arr_2;		//计算在参考图像中的映射坐标
				Mat diff = match2_xy_change - arr_1;

				pow(diff, 2, diff);

				//第一行和第二行求和，计算特征点间的距离的平方
				Mat add = diff(Range(0, 1), Range::all()) + diff(Range(1, 2), Range::all());

				float* p_add = add.ptr<float>(0);

				for (size_t i = 0; i < N; ++i)
				{
					//如果小于阈值
					if (p_add[i] < threshold)
					{
						right[i] = true;
						++consensus_num;
					}
					else
						right[i] = false;
				}
			}

			//判断当前一致集是否是优于之前最优一致集，并更新当前最优一致集个数
			if (consensus_num > most_consensus_num)
			{
				most_consensus_num = consensus_num;

				//把正确匹配的点赋予标签 1 
				for (size_t i = 0; i < N; ++i)
					inliers[i] = right[i];
			}
		}

		//删除重复点对
		for (size_t i = 0; i < N - 1; ++i)
		{
			for (size_t j = i + 1; j < N; ++j)
			{
				if (inliers[i] && inliers[j])
				{
					if (p10[i] == p10[j] && p11[i] == p11[j] && p20[i] == p20[j] && p21[i] == p21[j])
					{
						inliers[j] = false;
						--most_consensus_num;
					}
				}
			}
		}

		//迭代结束，获得最优一致集合，根据这些最优一致集合计算出最终的变换关系 T
		Mat consensus_arr1, consensus_arr2;				//经过迭代后最终确认正确匹配的点

		consensus_arr1.create(most_consensus_num, 2, CV_32FC1);
		consensus_arr2.create(most_consensus_num, 2, CV_32FC1);

		int k = 0;

		for (size_t i = 0; i < N; ++i)
		{
			if (inliers[i])
			{
				consensus_arr1.at<float>(k, 0) = p10[i];
				consensus_arr1.at<float>(k, 1) = p11[i];

				consensus_arr2.at<float>(k, 0) = p20[i];
				consensus_arr2.at<float>(k, 1) = p21[i];
				++k;
			}

		}

		int num_ransac = (model == string("similarity") ? 2 : (model == string("affine") ? 3 : 4));

		if (k < num_ransac)
			CV_Error(CV_StsBadArg, "ransac模块删除错误点对后剩下正确点对个数不足以计算出变换关系矩阵！");

		//利用迭代后正确匹配点计算变换矩阵，为什么不是挑选 n 个点计算变换矩阵
		T = LMSget(consensus_arr1, consensus_arr2, model, rmse);
		return T;
	}

	//生成网格图
	void myMatch::mosaic_map(const Mat& image_1, const Mat& image_2, Mat& chessboard_1, Mat& chessboard_2, Mat& mosaic_image, int width)
	{
		if (image_1.size != image_2.size)
			CV_Error(CV_StsBadArg, "mosaic_map模块输入两幅图大小必须一致！");

		//生成image_1的棋盘网格图
		chessboard_1 = image_1.clone();

		int rows_1 = chessboard_1.rows;
		int cols_1 = chessboard_1.cols;

		int row_grids_1 = cvFloor((double)rows_1 / width);		//行方向网格个数
		int col_grids_1 = cvFloor((double)cols_1 / width);		//列方向网格个数

		//指定区域像素赋值为零，便形成了棋盘图

		//第一幅图，第一行 2、4、6 像素值赋值零；第一幅图与第二幅图零像素位置交叉，以便两幅图交叉显示
		for (int i = 0; i < row_grids_1; i = i + 2)
		{
			for (int j = 1; j < col_grids_1; j = j + 2)
			{
				Range range_x(j * width, (j + 1) * width);
				Range range_y(i * width, (i + 1) * width);
				chessboard_1(range_y, range_x) = 0;
			}
		}

		for (int i = 1; i < row_grids_1; i = i + 2)
		{
			for (int j = 0; j < col_grids_1; j = j + 2)
			{
				Range range_x(j * width, (j + 1) * width);
				Range range_y(i * width, (i + 1) * width);

				chessboard_1(range_y, range_x) = 0;
			}
		}

		//生成image_2的棋盘网格图

		chessboard_2 = image_2.clone();

		int rows_2 = chessboard_2.rows;
		int cols_2 = chessboard_2.cols;

		int row_grids_2 = cvFloor((double)rows_2 / width);//行方向网格个数
		int col_grids_2 = cvFloor((double)cols_2 / width);//列方向网格个数

		//第二幅图，第一行 1、3、5 像素值赋值零
		for (int i = 0; i < row_grids_2; i = i + 2)
		{
			for (int j = 0; j < col_grids_2; j = j + 2)
			{
				Range range_x(j * width, (j + 1) * width);
				Range range_y(i * width, (i + 1) * width);
				chessboard_2(range_y, range_x) = 0;
			}
		}

		for (int i = 1; i < row_grids_2; i = i + 2)
		{
			for (int j = 1; j < col_grids_2; j = j + 2)
			{
				Range range_x(j * width, (j + 1) * width);
				Range range_y(i * width, (i + 1) * width);
				chessboard_2(range_y, range_x) = 0;
			}
		}

		//两个棋盘图进行叠加，显示配准效果
		mosaic_image = chessboard_1 + chessboard_2;
	}

	inline void median_filter(Mat& image, const vector<vector<int>>& pos)
	{
		int channels = image.channels();
		switch (channels)
		{
		case 1://单通道
			for (auto beg = pos.cbegin(); beg != pos.cend(); ++beg)
			{
				int i = (*beg)[0];//y
				int j = (*beg)[1];//x
				uchar& pix_val = image.at<uchar>(i, j);
				vector<uchar> pixs;
				for (int row = -1; row <= 1; ++row)
				{
					for (int col = -1; col <= 1; ++col)
					{
						if (i + row >= 0 && i + row < image.rows && j + col >= 0 && j + col < image.cols)
						{
							pixs.push_back(image.at<uchar>(i + row, j + col));
						}
					}
				}
				//排序
				std::sort(pixs.begin(), pixs.end());
				pix_val = pixs[pixs.size() / 2];
			}
			break;

		case 3://3通道
			for (auto beg = pos.cbegin(); beg != pos.cend(); ++beg)
			{
				int i = (*beg)[0];//y
				int j = (*beg)[1];//x
				Vec3b& pix_val = image.at<Vec3b>(i, j);
				vector<cv::Vec3b> pixs;
				for (int row = -1; row <= 1; ++row)
				{
					for (int col = -1; col <= 1; ++col)
					{
						if (i + row >= 0 && i + row < image.rows && j + col >= 0 && j + col < image.cols)
						{
							pixs.push_back(image.at<Vec3b>(i + row, j + col));
						}
					}
				}

				//排序
				std::sort(pixs.begin(), pixs.end(),
					[pix_val](const Vec3b& a, const Vec3b& b)->bool {
					return sum((a).ddot(a))[0] < sum((b).ddot(b))[0];
				});
				pix_val = pixs[pixs.size() / 2];
			}
			break;
		default:break;
		}
	}


	inline void min_dis_idx(const float* ptr_1, const Mat& des_2, int num_des2, int dims_des, float dis[2], int idx[2])
	{
		float min_dis1 = 1000, min_dis2 = 2000;
		int min_idx1, min_idx2;

		for (int j = 0; j < num_des2; ++j)
		{
			const float* ptr_des_2 = des_2.ptr<float>(j);
			float cur_dis = 0;
			for (int k = 0; k < dims_des; ++k)
			{
				float diff = ptr_1[k] - ptr_des_2[k];
				cur_dis += diff * diff;
			}
			if (cur_dis < min_dis1) {
				min_dis1 = cur_dis;
				min_idx1 = j;
			}
			else if (cur_dis >= min_dis1 && cur_dis < min_dis2) {
				min_dis2 = cur_dis;
				min_idx2 = j;
			}

		}
		dis[0] = sqrt(min_dis1); dis[1] = sqrt(min_dis2);
		idx[0] = min_idx1; idx[1] = min_idx2;
	}

	void myMatch::match_des(const Mat& des_1, const Mat& des_2, vector<vector<DMatch>>& dmatchs, DIS_CRIT dis_crite)
	{
		int num_des_1 = des_1.rows;
		int num_des_2 = des_2.rows;
		int dims_des = des_1.cols;

		vector<DMatch> match(2);

		//对于参考图像上的每一点，和待配准图像进行匹配
		if (dis_crite == 0)									//欧几里得距离
		{
			for (int i = 0; i < num_des_1; ++i)				//对于参考图像中的每个描述子
			{
				const float* ptr_des_1 = des_1.ptr<float>(i);
				float dis[2];
				int idx[2];
				min_dis_idx(ptr_des_1, des_2, num_des_2, dims_des, dis, idx);
				match[0].queryIdx = i;
				match[0].trainIdx = idx[0];
				match[0].distance = dis[0];

				match[1].queryIdx = i;
				match[1].trainIdx = idx[1];
				match[1].distance = dis[1];

				dmatchs.push_back(match);
			}
		}
		else if (dis_crite == 1)//cos距离
		{
			Mat trans_des2;
			transpose(des_2, trans_des2);
			double aa = (double)getTickCount();
			Mat mul_des = des_1 * trans_des2;
			//gemm(des_1, des_2, 1, Mat(), 0, mul_des, GEMM_2_T);
			double time1 = ((double)getTickCount() - aa) / getTickFrequency();
			std::cout << "cos距离中矩阵乘法花费时间： " << time1 << "s" << endl;

			for (int i = 0; i < num_des_1; ++i)
			{
				float max_cos1 = -1000, max_cos2 = -2000;
				int max_idx1, max_idx2;

				float* ptr_1 = mul_des.ptr<float>(i);
				for (int j = 0; j < num_des_2; ++j)
				{
					float cur_cos = ptr_1[j];
					if (cur_cos > max_cos1) {
						max_cos1 = cur_cos;
						max_idx1 = j;
					}
					else if (cur_cos <= max_cos1 && cur_cos > max_cos2) {
						max_cos2 = cur_cos;
						max_idx2 = j;
					}
				}

				match[0].queryIdx = i;
				match[0].trainIdx = max_idx1;
				match[0].distance = acosf(max_cos1);

				match[1].queryIdx = i;
				match[1].trainIdx = max_idx2;
				match[1].distance = acosf(max_cos2);
				dmatchs.push_back(match);
			}
		}
	}

	void myMatch::scale_ROM_Histogram(const vector<DMatch>& matches, float* scale_hist, float* ROM_hist, int n)
	{
		int len = matches.size();

		//使用AutoBuffer分配一段内存，这里多出4个空间的目的是为了方便后面平滑直方图的需要
		AutoBuffer<float> buffer((4 * len) + n + 4);

		//X保存水平差分，Y保存数值差分，Mag保存梯度幅度，Ori保存梯度角度，W保存高斯权重
		float* X = buffer, * Y = buffer + len, * Mag = Y, * Ori = Y + len, * W = Ori + len;
		float* temp_hist = W + len + 2;						//临时保存直方图数据

		for (int i = 0; i < n; ++i)
			temp_hist[i] = 0.f;								//数据清零
	}

	Mat myMatch::match(const Mat& image_1, const Mat& image_2, const vector<vector<DMatch>> dmatchs, vector<KeyPoint> keys_1,
		vector<KeyPoint> keys_2, string model, vector<vector<DMatch>>& dmatchs2, vector<DMatch>& right_matchs, Mat& matched_line, vector<DMatch>& init_matchs,
		vector<Point2f>& point_11, vector<Point2f>& point_22, const Mat& des_1, const Mat& des_2)
	{
		//获取初始匹配的关键点的位置
		vector<Point2f> point_1, point_2;
		int num = 0, sum = 0;
		vector<DMatch> error_matchs;
		for (size_t i = 0; i < dmatchs.size(); ++i)						//增加一个一个0.8，正确匹配点数增加2
		{
			double dis_1 = dmatchs[i][0].distance;						//distance对应的是特征点描述符的欧式距离
			double dis_2 = dmatchs[i][1].distance;
			int idx1 = dmatchs[i][0].queryIdx;
			int idx2 = dmatchs[i][0].trainIdx;
			//cout << dis_1 / dis_2 << endl;
			//比率测试筛选误匹配点(初步筛选)，如果满足则认为是有候选集中有正确匹配点
			if (dis_1 < dis_ratio * dis_2 )						//最近邻和次近邻距离比阈值
			{
				/*
				for (int j = 0; j < 128; j++)
				{
					cout << des_1.at<float>(idx1, j) << "  ";
				}
				cout << endl;
				for (int j = 0; j < 128; j++)
				{
					cout << des_2.at<float>(idx2, j) << "  ";
				}
				cout << endl;
				*/
				//queryIdx、trainIdx和distance是DMatch类中的一些属性	//pt是KeyPoint类中的成员，对应关键点的坐标
				//for (int j = 0; j < dmatchs2.size(); j++)
				{
					
					//if (dmatchs2[j][0].queryIdx == idx2 && dmatchs2[j][0].trainIdx == idx1 )
					{
						float dis = pow((keys_1[idx1].pt.x - keys_2[idx2].pt.x), 2) + pow((keys_1[idx1].pt.y - keys_2[idx2].pt.y), 2);
						point_11.push_back(keys_1[idx1].pt);	//queryIdx对应的是特征描述子的下标，也是对应特征点的下标
						point_22.push_back(keys_2[idx2].pt);	//trainIdx对应的是特征描述子的下标，也是对应特征点的下标
						//cout << keys_1[idx1].pt.x << "   " << keys_2[idx2].pt.x << "    " << keys_1[idx1].pt.y << "   " << keys_2[idx2].pt.y << endl;
						init_matchs.push_back(dmatchs[i][0]);					//保存正确的dmatchs
					}
				}
			}
			else if ((dis_1 / dis_2) > 0.99)
			{
				error_matchs.push_back(dmatchs[i][0]);
			}
		}


		std::cout << "距离比之后初始匹配点对个数是： " << init_matchs.size() << endl;

		int min_pairs = (model == string("similarity") ? 2 : (model == string("affine") ? 3 : 4));

		if (point_11.size() < min_pairs)
			CV_Error(CV_StsBadArg, "match模块距离比阶段匹配特征点个数不足！");

		////使用ransac算法再次对匹配点进行筛选，然后使用最后确定的匹配点计算变换矩阵的参数
		//vector<bool> inliers;											//存放的是 bool 类型的数据，对应特征点
		//float rmse;

		////homography是一个(3,3)矩阵，是待配准影像到参考影像的变换矩阵，初始误差阈值是 1.5
		Mat homography;
		////homography = ransac(point_1, point_2, model, 1.5, inliers, rmse);
		////cout << "ransac finished" << endl;
		////提取出处正确匹配点对
		//int i = 0;
		//vector<DMatch>::iterator itt = init_matchs.begin();
		//for (vector<bool>::iterator it = inliers.begin(); it != inliers.end(); ++it, ++itt)
		//{
		//	if (*it)								//如果是正确匹配点对
		//	{
		//		right_matchs.push_back(*itt);
		//		// init_matchs 中匹配点对的存储顺序和 point_1 中特征点的存储顺序是一一对应的
		//		point_11.push_back(point_1.at(i));
		//		point_22.push_back(point_2.at(i));
		//	}
		//	++i;

		//}
		//cout << "使用RANSAC删除错误点对,且返回正确匹配个数： " << right_matchs.size() << endl;
		//cout << "误差rmse: " << rmse << endl;


		Mat error_matched;													//输出矩阵，类似画布
		cv::drawMatches(image_1, keys_1, image_2, keys_2, error_matchs, error_matched,
			 Scalar(0, 255, 255), Scalar(139, 0, 0), vector<char>());	//该函数用于绘制特征点并对匹配的特征点进行连线
		for (int i = 0; i < error_matchs.size(); ++i) {
			cv::DMatch match = error_matchs[i];
			cv::Point2f pt1 = keys_1[match.queryIdx].pt;
			cv::Point2f pt2 = keys_2[match.trainIdx].pt;
			pt2.x += 512;
			cv::line(error_matched, pt1, pt2, Scalar(0, 255, 255), 1);
		}
		cv::imwrite("D://OpenCorr_VS2019/gauss/错误匹配点对.jpg", error_matched);

		//绘制初始匹配点对连线图,此时初始匹配指的是经过 KnnMatch 筛选后的匹配
		Mat initial_matched;													//输出矩阵，类似画布
		cv::drawMatches(image_1, keys_1, image_2, keys_2, init_matchs, initial_matched,
			Scalar(255, 0, 255), Scalar(0, 255, 0), vector<char>());	//该函数用于绘制特征点并对匹配的特征点进行连线
		cv::imwrite("D://OpenCorr_VS2019/gauss/初始匹配点对.jpg", initial_matched);			//保存图片,第一个颜色控制连线，第二个颜色控制特征点
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(init_matchs.begin(), init_matchs.end(), g);
		vector<DMatch> draws(init_matchs.begin(), init_matchs.begin() + 100);
		//绘制正确匹配点对连线图
		cv::drawMatches(image_1, keys_1, image_2, keys_2, draws, matched_line,
			Scalar(0, 255, 255), Scalar(139, 0, 0), vector<char>());
		// 调整画笔宽度
		for (int i = 0; i < draws.size(); ++i) {
			cv::DMatch match = draws[i];
			cv::Point2f pt1 = keys_1[match.queryIdx].pt;
			cv::Point2f pt2 = keys_2[match.trainIdx].pt;
			pt2.x += 512;
			cv::line(matched_line, pt1, pt2, cv::Scalar(0, 255, 255), 2); 
		}
		cv::imwrite("D://OpenCorr_VS2019/gauss/正确匹配点对.jpg", matched_line);

		//保存和显示检测到的特征点
		Mat keys_image_1, keys_image_2;											//输出矩阵，类似画布
		cv::drawKeypoints(image_1, keys_1, keys_image_1, Scalar(0, 255, 0));		//该函数用于绘制图像中的特征点
		cv::drawKeypoints(image_2, keys_2, keys_image_2, Scalar(0, 255, 0));
		cv::imwrite("D://OpenCorr_VS2019/gauss/参考图像检测到的特征点.jpg", keys_image_1);
		cv::imwrite("D://OpenCorr_VS2019/gauss/待配准图像检测到的.jpg", keys_image_2);

		return homography;
	}
}