#include "my_sift.h"

#include<string>
#include<cmath>
#include<iostream>			//输入输出
#include<vector>			//vector
#include<algorithm>
#include<numeric>			//用于容器元素求和

#include<opencv2\core\hal\hal.hpp>
#include<opencv2\core\hal\intrin.hpp>
#include <fstream>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>				//opencv基本数据结构
#include<opencv2\highgui\highgui.hpp>		//图像界面
#include<opencv2\imgproc\imgproc.hpp>		//基本图像处理函数
#include<opencv2\features2d\features2d.hpp> //特征提取
#include<opencv2\imgproc\types_c.h>
#include<opencv2\videoio.hpp>

using namespace std;
using namespace cv;

namespace opencorr
{
	vector<vector<float> > MySift::get_degree() const
	{
		vector<Point2f> points;
		for (int i = 1; i <= 4; i++)
		{
			for (int j = 1; j <= 4; j++)
			{
				Point2f point(i, j);
				points.push_back(point);
			}
		}
		Point2f centerpoint(2.5, 2.5);
		vector<vector<float> > degree;
		vector<float> add_degree = { 270, 270, 270, 270, 270, 270, 270, 270, -90, -90, 270, 270, -90, -90, 270, 270 };
		for (int i = 0; i < points.size(); i++)
		{
			float deg = atan2(centerpoint.y - points[i].y, centerpoint.x - points[i].x) * 180.0 / CV_PI;
			deg += add_degree[i];
			vector<float> temp;
			temp.push_back(deg);
			double deg1 = deg + 120;
			if (deg1 >= 360)
			{
				deg1 -= 360;
			}
			double deg2 = deg + 240;
			if (deg2 >= 360)
			{
				deg2 -= 360;
			}
			temp.push_back(deg1);
			temp.push_back(deg2);
			sort(temp.begin(), temp.end());
			degree.push_back(temp);
		}

		return degree;
	}
	int MySift::num_octaves(const Mat& image) const
	{
		int temp;
		float size_temp = (float)min(image.rows, image.cols);
		temp = cvRound(log(size_temp) / log((float)2) - 2);

		if (double_size)
			temp += 1;
		if (temp > MAX_OCTAVES)//尺度空间最大组数设置为MAX_OCTAVES
			temp = MAX_OCTAVES;

		return temp;
	}

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

	// 定义SIFT标准宏（避免未定义）
#ifndef DESCR_SCL_FCTR
#define DESCR_SCL_FCTR 1.5f
#endif

// 函数声明（兼容原接口）
	static void new2_calc_sift_descriptor(const cv::Mat& gauss_image, float main_angle, cv::Point2f pt,
		float scale, int d, int n, float* descriptor,
		std::vector<std::vector<float>> /*degree*/, int length);

	// 辅助函数：镜像填充边界像素（解决大旋转下边界梯度丢失）
	static cv::Point get_mirrored_coord(int x, int y, int cols, int rows) {
		// 行方向镜像
		if (y < 0) y = -y - 1;
		else if (y >= rows) y = 2 * rows - y - 1;
		// 列方向镜像
		if (x < 0) x = -x - 1;
		else if (x >= cols) x = 2 * cols - x - 1;
		return cv::Point(x, y);
	}

	// 辅助函数：计算Cell局部主方向（用于Bin方向微调）
	static float get_cell_local_angle(const cv::Mat& gauss_image, const cv::Rect& cell_rect) {
		float max_mag = 0.0f;
		float local_angle = 0.0f;
		int cols = gauss_image.cols;
		int rows = gauss_image.rows;

		for (int y = 0; y < cell_rect.height; ++y) {
			for (int x = 0; x < cell_rect.width; ++x) {
				// 镜像处理边界像素
				cv::Point coord = get_mirrored_coord(cell_rect.x + x, cell_rect.y + y, cols, rows);
				int r = coord.y;
				int c = coord.x;

				// 计算梯度（2像素间隔，保留原精度）
				float dx = 0.5f * (gauss_image.at<float>(r, cv::min(c + 1, cols - 1)) - gauss_image.at<float>(r, cv::max(c - 1, 0)));
				float dy = 0.5f * (gauss_image.at<float>(cv::min(r + 1, rows - 1), c) - gauss_image.at<float>(cv::max(r - 1, 0), c));
				float mag = sqrt(dx * dx + dy * dy);

				if (mag > max_mag) {
					max_mag = mag;
					local_angle = cv::fastAtan2(dy, dx); // 0-360°
				}
			}
		}
		return local_angle;
	}

	static void new2_calc_sift_descriptor(const Mat& gauss_image, float main_angle, Point2f pt,
		float scale, int d, int n, float* descriptor, vector<vector<float>> /*degree*/, int length)
	{
		// 恢复严格的角度区间（120°等分，无扩大）
		vector<vector<float>> dynamic_degree(d * d, vector<float>(3));
		for (int cell_id = 0; cell_id < d * d; ++cell_id) {
			float a0 = fmod(main_angle, 360.0f);
			float a1 = fmod(main_angle + 120.0f, 360.0f);
			float a2 = fmod(main_angle + 240.0f, 360.0f);

			if (a1 < a0) a1 += 360.0f;
			if (a2 < a1) a2 += 360.0f;
			dynamic_degree[cell_id][0] = a0;
			dynamic_degree[cell_id][1] = a1;
			dynamic_degree[cell_id][2] = a2;
		}

		Point ptxy(cvRound(pt.x), cvRound(pt.y));
		float cos_t = cosf(-main_angle * (float)(CV_PI / 180));
		float sin_t = sinf(-main_angle * (float)(CV_PI / 180));
		float bins_per_rad = n / 360.f;
		float exp_scale = -1.f / (d * d * 0.5f);  // 恢复高斯权重衰减速度（过滤远处噪声）
		float hist_width = DESCR_SCL_FCTR * scale;  // 恢复原始直方图宽度
		int radius = cvRound(hist_width * (d + 1) * sqrt(2) * 0.5f);
		int rows = gauss_image.rows, cols = gauss_image.cols;

		radius = min(radius, (int)sqrt((double)rows * rows + cols * cols));

		cos_t = cos_t / hist_width;
		sin_t = sin_t / hist_width;

		int len = (2 * radius + 1) * (2 * radius + 1);
		int histlen = (d + 2) * (d + 2) * (n + 2);

		AutoBuffer<float> buf(6 * len + histlen);

		float* X = buf, * Y = buf + len, * Mag = Y, * Angle = Y + len, * W = Angle + len;
		float* RBin = W + len, * CBin = RBin + len, * hist = CBin + len;

		memset(hist, 0, histlen * sizeof(float));

		// 提高梯度阈值（过滤低对比度噪声）
		int k = 0;
		float max_mag = 0.0f;
		for (int i = -radius; i < radius; ++i)
		{
			for (int j = -radius; j < radius; ++j)
			{
				float c_rot = j * cos_t - i * sin_t;
				float r_rot = j * sin_t + i * cos_t;

				float cbin = c_rot + d / 2 - 0.5f;
				float rbin = r_rot + d / 2 - 0.5f;

				float r_float = pt.y + i;
				float c_float = pt.x + j;
				int r = cvRound(r_float);
				int c = cvRound(c_float);

				// 收紧边界条件，避免边缘噪声
				if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
					r > 1 && r < rows - 2 && c > 1 && c < cols - 2)  // 远离边界1像素
				{
					// 梯度计算（仅用内部像素，确保精度）
					float dx = 0.5f * (gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1));
					float dy = 0.5f * (gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c));

					float mag = sqrt(dx * dx + dy * dy);
					if (mag > max_mag) max_mag = mag;

					X[k] = dx;
					Y[k] = dy;
					CBin[k] = cbin;
					RBin[k] = rbin;
					W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;
					k++;
				}
			}
		}

		// 恢复5%的梯度阈值，过滤低对比度噪声
		len = k;
		float mag_thresh = (max_mag > 0) ? 0.05f * max_mag : 0;
		int valid_len = 0;
		for (int i = 0; i < len; ++i) {
			float mag = sqrt(X[i] * X[i] + Y[i] * Y[i]);
			if (mag >= mag_thresh) {  // 不强制保留低质量特征
				X[valid_len] = X[i];
				Y[valid_len] = Y[i];
				CBin[valid_len] = CBin[i];
				RBin[valid_len] = RBin[i];
				W[valid_len] = W[i];
				valid_len++;
			}
		}
		len = valid_len;

		// 计算梯度幅度、角度和权重
		cv::hal::exp(W, W, len);
		cv::hal::fastAtan2(Y, X, Angle, len, true);
		cv::hal::magnitude(X, Y, Mag, len);

		// 增强方向信息的区分度
		for (k = 0; k < len; ++k)
		{
			float rbin = RBin[k], cbin = CBin[k];
			float temp_angle = Angle[k];
			float mag = Mag[k] * W[k];
			int r0 = cvFloor(rbin);
			int c0 = cvFloor(cbin);

			rbin = rbin - r0;
			cbin = cbin - c0;

			// 增强方向权重（差异大的特征显著降权）
			float angle_diff = fabs(fmod(temp_angle - main_angle + 360, 360));
			angle_diff = min(angle_diff, 360 - angle_diff);
			float dir_weight = exp(-(angle_diff * angle_diff) / (60.0f * 60.0f));  // 高斯衰减（60°为半宽）
			mag *= dir_weight;

			// 四个角落的权重计算（保留高精度插值）
			vector<int> ids = { r0 * 4 + c0, r0 * 4 + c0 + 1,
							   (r0 + 1) * 4 + c0, (r0 + 1) * 4 + c0 + 1 };

			for (int idx = 0; idx < 4; ++idx) {
				int id = ids[idx];
				if (id < 0 || id >= d * d) continue;

				float a0 = dynamic_degree[id][0];
				float a1 = dynamic_degree[id][1];
				float a2 = dynamic_degree[id][2];

				float norm_angle = temp_angle;
				if (norm_angle < a0) norm_angle += 360.0f;

				int o0 = -1;
				float obin = 0.0f;
				if (norm_angle < a1) {
					o0 = 0;
					obin = (norm_angle - a0) / (a1 - a0);
				}
				else if (norm_angle < a2) {
					o0 = 1;
					obin = (norm_angle - a1) / (a2 - a1);
				}
				else {
					o0 = 2;
					obin = (norm_angle - a2) / (a0 + 360.0f - a2);
				}

				if (o0 == -1) continue;

				float v_rc = 0.0f;
				switch (idx) {
				case 0: v_rc = (1 - rbin) * (1 - cbin); break;
				case 1: v_rc = (1 - rbin) * cbin; break;
				case 2: v_rc = rbin * (1 - cbin); break;
				case 3: v_rc = rbin * cbin; break;
				}
				float weighted_mag = mag * v_rc;

				int hist_r = r0 + 1 + (idx >= 2 ? 1 : 0);
				int hist_c = c0 + 1 + (idx % 2 == 1 ? 1 : 0);
				int hist_idx = ((hist_r * (d + 2) + hist_c) * (n + 2)) + o0;

				hist[hist_idx] += weighted_mag * (1 - obin);
				hist[hist_idx + 1] += weighted_mag * obin;
			}
		}

		// 调整直方图（保留角度区分度）
		for (int i = 0; i < d; ++i)
		{
			for (int j = 0; j < d; ++j)
			{
				int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
				hist[idx] += hist[idx + n];  // 正常合并跨边界角度

				for (int k = 0; k < n; ++k) {
					descriptor[(i * d + j) * n + k] = hist[idx + k];
				}
			}
		}

		// 恢复角度+模长排序（增强方向一致性）
		for (int i = 0; i < d; ++i)
		{
			for (int j = 0; j < d; ++j)
			{
				int base_idx = (i * d + j) * n;
				vector<pair<float, float>> vecs;
				for (int b = 0; b < n; ++b)
				{
					float mag = descriptor[base_idx + b];
					float angle = dynamic_degree[i * d + j][b];
					vecs.emplace_back(mag, angle);
				}

				// 优先按角度与主方向的一致性排序，再按模长
				sort(vecs.begin(), vecs.end(), [&](const pair<float, float>& a, const pair<float, float>& b) {
					float diff_a = fabs(fmod(a.second - main_angle + 360, 360));
					diff_a = min(diff_a, 360 - diff_a);

					float diff_b = fabs(fmod(b.second - main_angle + 360, 360));
					diff_b = min(diff_b, 360 - diff_b);

					if (fabs(diff_a - diff_b) > 1e-3) {
						return diff_a < diff_b;
					}
					return a.first > b.first;
					});

				for (int b = 0; b < n; ++b)
				{
					descriptor[base_idx + b] = vecs[b].first;
				}

				// 叉积特征仅保留最显著的一个（增强稳定性）
				if (n >= 3)
				{
					float Fl = vecs[0].first;
					float Fm = vecs[1].first;

					float thetal = vecs[0].second;
					float thetam = vecs[1].second;

					float delta_lm = (thetal - thetam);
					delta_lm = fmod(delta_lm + 360, 360);
					if (delta_lm > 180) delta_lm -= 360;
					delta_lm *= CV_PI / 180.0f;

					float D1 = Fl * Fm * sin(delta_lm);
					descriptor[d * d * n + i * d + j] = D1;
				}
			}
		}

		// 恢复L2归一化（抑制噪声更有效）
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				int base_idx = (i * d + j) * n;
				float cell_norm = 0;
				for (int b = 0; b < n; ++b) {
					cell_norm += descriptor[base_idx + b] * descriptor[base_idx + b];
				}
				cell_norm = sqrt(cell_norm) + 1e-8;
				for (int b = 0; b < n; ++b) {
					descriptor[base_idx + b] /= cell_norm;
				}
			}
		}

		// 收紧截断阈值（0.2），抑制异常大值
		int vec_len = d * d * n;
		float trunc_thresh = 0.2f;
		for (int i = 0; i < vec_len; ++i)
		{
			descriptor[i] = min(descriptor[i], trunc_thresh);
		}
	}


	static void new_calc_sift_descriptor(const Mat& gauss_image, float main_angle, Point2f pt,
		float scale, int d, int n, float* descriptor, vector<vector<float>> degree, int length)
	{
		// d = 8
		Point ptxy(cvRound(pt.x), cvRound(pt.y));					//坐标取整
		float cos_t = cosf(-main_angle * (float)(CV_PI / 180));		//把角度转化为弧度，计算主方向的余弦
		float sin_t = sinf(-main_angle * (float)(CV_PI / 180));		//把角度转化为弧度，计算主方向的正弦
		float bins_per_rad = n / 360.f;								// n = 8 ，梯度直方图分为 8 个方向
		float exp_scale = -1.f / (d * d * 0.5f);					//权重指数部分
		float hist_width = DESCR_SCL_FCTR * scale;					//特征点邻域内子区域边长，子区域的边长
		int radius = cvRound(hist_width * (d + 1) * sqrt(2) * 0.5f);//特征点邻域半径(d+1)*(d+1)，四舍五入
		int rows = gauss_image.rows, cols = gauss_image.cols;		//当前高斯层行、列信息

		//特征点邻域半径
		radius = min(radius, (int)sqrt((double)rows * rows + cols * cols));

		cos_t = cos_t / hist_width;
		sin_t = sin_t / hist_width;

		int len = (2 * radius + 1) * (2 * radius + 1);				//邻域内总像素数，为后面动态分配内存使用
		int histlen = (d + 2) * (d + 2) * (n + 2);					//值为 360 

		AutoBuffer<float> buf(6 * len + histlen);

		//X保存水平差分，Y保存竖直差分，Mag保存梯度幅度，Angle保存特征点方向, W保存高斯权重
		float* X = buf, * Y = buf + len, * Mag = Y, * Angle = Y + len, * W = Angle + len;
		float* RBin = W + len, * CBin = RBin + len, * hist = CBin + len;

		//首先清空直方图数据
		for (int i = 0; i < d + 2; ++i)				// i 对应 row
		{
			for (int j = 0; j < d + 2; ++j)			// j 对应 col
			{
				for (int k = 0; k < n + 2; ++k)

					hist[(i * (d + 2) + j) * (n + 2) + k] = 0.f;
			}
		}

		//把邻域内的像素分配到相应子区域内，计算子区域内每个像素点的权重(子区域即 d*d 中每一个小方格)
		int k = 0;

		//实际上是在 4 x 4 的网格中找 16 个种子点，每个种子点都在子网格的正中心，
		//通过三线性插值对不同种子点间的像素点进行加权作用到不同的种子点上绘制方向直方图

		int minx = 0, miny = 0;
		for (int i = -radius; i < radius; ++i)						// i 对应 y 行坐标
		{
			for (int j = -radius; j < radius; ++j)					// j 对应 x 列坐标
			{
				float c_rot = j * cos_t - i * sin_t;				//旋转后邻域内采样点的 x 坐标
				float r_rot = j * sin_t + i * cos_t;				//旋转后邻域内采样点的 y 坐标

				//旋转后 5 x 5 的网格中的所有像素点被分配到 4 x 4 的网格中
				float cbin = c_rot + d / 2 - 0.5f;					//旋转后的采样点落在子区域的 x 坐标
				float rbin = r_rot + d / 2 - 0.5f;					//旋转后的采样点落在子区域的 y 坐标

				int r = ptxy.y + i, c = ptxy.x + j;					//ptxy是高斯金字塔中的坐标

				//这里rbin，cbin范围是(-1,d)
				if (rbin > -1 && rbin < d && cbin>-1 && cbin < d && r>0 && r < rows - 1 && c>0 && c < cols - 1)
				{
					float dx = gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
					float dy = gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);

					X[k] = dx;												//邻域内所有像素点的水平差分
					Y[k] = dy;												//邻域内所有像素点的竖直差分

					CBin[k] = cbin;											//邻域内所有采样点落在子区域的 x 坐标
					RBin[k] = rbin;											//邻域内所有采样点落在子区域的 y 坐标

					W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;		//高斯权值的指数部分
					k++;
				}
			}
		}

		//计算采样点落在子区域的像素梯度幅度，梯度角度，和高斯权值
		len = k;
		cv::hal::exp(W, W, len);						//邻域内所有采样点落在子区域的像素的高斯权值
		cv::hal::fastAtan2(Y, X, Angle, len, true);		//邻域内所有采样点落在子区域的像素的梯度方向，角度范围是0-360度
		cv::hal::magnitude(X, Y, Mag, len);				//邻域内所有采样点落在子区域的像素的梯度幅度
		//vector<vector<float> > hist(8,vector<float>(8,0));
		//实际上是在 4 x 4 的网格中找 16 个种子点，每个种子点都在子网格的正中心，
		//通过三线性插值对不同种子点间的像素点进行加权作用到不同的种子点上绘制方向直方图
		//计算角度

		//计算每个特征点的特征描述子
		for (k = 0; k < len; ++k)
		{
			float rbin = RBin[k], cbin = CBin[k];		//子区域内像素点坐标，rbin,cbin范围是(-1,d)
			//子区域内像素点处理后的方向
			//float temp = Angle[k] - main_angle;
			//if (temp < 0)
			//{
			//	temp += 360;
			//}
			float temp = Angle[k];
			float obin = temp * bins_per_rad;			//指定方向的数量后，邻域内像素点对应的方向
			float mag = Mag[k] * W[k];					//子区域内像素点乘以权值后的梯度幅值
			int r0 = cvFloor(rbin);						//ro取值集合是{-1，0，1，2，3}
			int c0 = cvFloor(cbin);						//c0取值集合是{-1，0，1，2，3}
			int o0 = cvFloor(obin);

			rbin = rbin - r0;							//子区域内像素点坐标的小数部分，用于线性插值，分配像素点的作用
			cbin = cbin - c0;
			obin = obin - o0;							//子区域方向的小数部分

			//限制范围为梯度直方图横坐标[0,n)，8 个方向直方图
			if (o0 < 0)
				o0 = o0 + n;
			if (o0 >= n)
				o0 = o0 - n;

			//三线性插值用于计算落在两个子区域之间的像素对两个子区域的作用，并把其分配到对应子区域的8个方向上
			//像素对应的信息通过加权分配给其周围的种子点，并把相应方向的梯度值进行累加  

			float v_r1 = mag * rbin;					//第二行分配的值
			float v_r0 = mag - v_r1;					//第一行分配的值


			float v_rc11 = v_r1 * cbin;					//第二行第二列分配的值，右下角种子点
			float v_rc10 = v_r1 - v_rc11;				//第二行第一列分配的值，左下角种子点

			float v_rc01 = v_r0 * cbin;					//第一行第二列分配的值，右上角种子点
			float v_rc00 = v_r0 - v_rc01;				//第一行第一列分配的值，左上角种子点


			//一个像素点的方向为每个种子点的两个方向做出贡献
			float v_rco111 = v_rc11 * obin;				//右下角种子点第二个方向上分配的值
			float v_rco110 = v_rc11 - v_rco111;			//右下角种子点第一个方向上分配的值

			float v_rco101 = v_rc10 * obin;           //左下角
			float v_rco100 = v_rc10 - v_rco101;

			float v_rco011 = v_rc01 * obin;           //右上角
			float v_rco010 = v_rc01 - v_rco011;

			float v_rco001 = v_rc00 * obin;           //左上角
			float v_rco000 = v_rc00 - v_rco001;

			float temp_obin = obin;
			//该像素所在网格的索引
			int id = r0 * 4 + c0;//左上
			//cout << id << endl;
			o0 = -1;
			if (id < 0 || id > 15)
			{
				o0 = -1;
				obin = temp_obin;
			}
			else if (temp < degree[id][0])
			{
				o0 = 2;
				obin = (temp + 360 - degree[id][2]) * bins_per_rad;
			}
			else if (temp >= degree[id][0] && temp < degree[id][1])
			{
				o0 = 0;
				obin = (temp - degree[id][0]) * bins_per_rad;
			}
			else if (temp >= degree[id][1] && temp < degree[id][2])
			{
				o0 = 1;
				obin = (temp - degree[id][1]) * bins_per_rad;
			}
			else if (temp >= degree[id][2])
			{
				o0 = 2;
				obin = (temp - degree[id][2]) * bins_per_rad;
			}
			if (obin < 0 || obin>1)
			{
				cout << o0 << "  " << obin << endl;
			}
			//o0 = getdegreeid(id,temp,degree);
			int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
			v_rco001 = v_rc00 * obin;           //左上角
			v_rco000 = v_rc00 - v_rco001;
			hist[idx] += v_rco000;
			hist[idx + 1] += v_rco001;

			id = id + 1;
			//o0 = getdegreeid(id+1, temp,degree);//右上
			o0 = -1;
			if (id < 0 || id>15)
			{
				o0 = -1;
				obin = temp_obin;
			}
			else if (temp < degree[id][0])
			{
				o0 = 2;
				obin = (temp + 360 - degree[id][2]) * bins_per_rad;
			}
			else if (temp >= degree[id][0] && temp < degree[id][1])
			{
				o0 = 0;
				obin = (temp - degree[id][0]) * bins_per_rad;
			}
			else if (temp >= degree[id][1] && temp < degree[id][2])
			{
				o0 = 1;
				obin = (temp - degree[id][1]) * bins_per_rad;
			}
			else if (temp >= degree[id][2])
			{
				o0 = 2;
				obin = (temp - degree[id][2]) * bins_per_rad;
			}
			if (obin < 0 || obin>1)
			{
				cout << o0 << "  " << obin << endl;
			}
			idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
			v_rco011 = v_rc01 * obin;           //右上角
			v_rco010 = v_rc01 - v_rco011;
			hist[idx + n + 2] += v_rco010;
			hist[idx + n + 3] += v_rco011;

			id = id + 3;
			//o0 = getdegreeid(id + 4, temp,degree);//左下
			o0 = -1;
			if (id < 0 || id>15)
			{
				o0 = -1;
				obin = temp_obin;
			}
			else if (temp < degree[id][0])
			{
				o0 = 2;
				obin = (temp + 360 - degree[id][2]) * bins_per_rad;
			}
			else if (temp >= degree[id][0] && temp < degree[id][1])
			{
				o0 = 0;
				obin = (temp - degree[id][0]) * bins_per_rad;
			}
			else if (temp >= degree[id][1] && temp < degree[id][2])
			{
				o0 = 1;
				obin = (temp - degree[id][1]) * bins_per_rad;
			}
			else if (temp >= degree[id][2])
			{
				o0 = 2;
				obin = (temp - degree[id][2]) * bins_per_rad;
			}
			if (obin < 0 || obin>1)
			{
				cout << temp << "  " << degree[id][0] << "  " << o0 << "  " << obin << endl;
			}
			idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
			v_rco101 = v_rc10 * obin;           //左下角
			v_rco100 = v_rc10 - v_rco101;
			hist[idx + (d + 2) * (n + 2)] += v_rco100;
			hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;

			id = id + 1;
			//o0 = getdegreeid(id + 5, temp,degree);//左下
			o0 = -1;
			if (id < 0 || id>15)
			{
				o0 = -1;
				obin = temp_obin;
			}
			if (id < 0 || id>15)
			{
				o0 = -1;
			}
			else if (temp < degree[id][0])
			{
				o0 = 2;
				obin = (temp + 360 - degree[id][2]) * bins_per_rad;
			}
			else if (temp >= degree[id][0] && temp < degree[id][1])
			{
				o0 = 0;
				obin = (temp - degree[id][0]) * bins_per_rad;
			}
			else if (temp >= degree[id][1] && temp < degree[id][2])
			{
				o0 = 1;
				obin = (temp - degree[id][1]) * bins_per_rad;
			}
			else if (temp >= degree[id][2])
			{
				o0 = 2;
				obin = (temp - degree[id][2]) * bins_per_rad;
			}
			if (obin < 0 || obin>1)
			{
				cout << o0 << "  " << obin << endl;
			}
			idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
			v_rco111 = v_rc11 * obin;				//右下角种子点第二个方向上分配的值
			v_rco110 = v_rc11 - v_rco111;
			hist[idx + (d + 3) * (n + 2)] += v_rco110;
			hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;

			/*
			hist[idx] += v_rco000;
			hist[idx + 1] += v_rco001;
			hist[idx + n + 2] += v_rco010;
			hist[idx + n + 3] += v_rco011;
			hist[idx + (d + 2) * (n + 2)] += v_rco100;
			hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
			hist[idx + (d + 3) * (n + 2)] += v_rco110;
			hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
			*/
		}

		//由于圆周循环的特性，对计算以后幅角小于 0 度或大于 360 度的值重新进行调整，使
		//其在 0～360 度之间
		//float desc[128];
		for (int i = 0; i < d; ++i)
		{
			for (int j = 0; j < d; ++j)
			{
				//类似于 hist[0][2][3] 第 0 行，第 2 列，种子点直方图中的第 3 个 bin
				int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);

				hist[idx] += hist[idx + n];
				//hist[idx + 1] += hist[idx + n + 1];//opencv源码中这句话是多余的,hist[idx + n + 1]永远是0.0

				for (k = 0; k < n; ++k)
					descriptor[(i * d + j) * n + k] = hist[idx + k];
			}
		}



		//对描述子进行归一化
		//int lenght = d * d * n;
		int lenght = d * d * (n + 0);
		float norm = 0;

		//计算特征描述向量的模值的平方
		for (int i = 0; i < lenght; ++i)
		{
			norm = norm + descriptor[i] * descriptor[i];
		}
		norm = sqrt(norm);							//特征描述向量的模值

		//此次归一化能去除光照的影响
		for (int i = 0; i < lenght; ++i)
		{
			descriptor[i] = descriptor[i] / norm;
		}

		//阈值截断,去除特征描述向量中大于 0.2 的值，能消除非线性光照的影响(相机饱和度对某些放的梯度影响较大，对方向的影响较小)
		for (int i = 0; i < lenght; ++i)
		{
			descriptor[i] = min(descriptor[i], DESCR_MAG_THR);
		}


		//再次归一化，能够提高特征的独特性
		norm = 0;
		for (int i = 0; i < lenght; ++i)
		{
			norm = norm + descriptor[i] * descriptor[i];
		}
		norm = sqrt(norm);
		for (int i = 0; i < lenght; ++i)
		{
			descriptor[i] = descriptor[i] / norm;
		}


		int idx = d * d * n;
		float strength_max = 0, strength_min = 100;
		for (int i = 0; i < d * d * n; i += 3)
		{
			float a = descriptor[i], b = descriptor[i + 1], c = descriptor[i + 2];
			/*
			float strength = pow(a * b * sin(120 * CV_PI / 180), 1.0 / 1);
			strength*= pow(a * c * sin(120 * CV_PI / 180),1.0/1);
			strength *=pow( b * c * sin(120 * CV_PI / 180),1.0/1);
			strength_max = max(strength_max, strength);
			strength_min = min(strength_min, strength);
			*/
			float strength1 = a * b * sin(120 * CV_PI / 180);
			float strength2 = a * c * sin(120 * CV_PI / 180);
			float strength3 = b * c * sin(120 * CV_PI / 180);
			strength_max = max(strength_max, max(max(strength1, strength2), strength3));
			strength_min = min(strength_min, min(min(strength1, strength2), strength3));
			//	strength_max = max(strength1, strength_max);
			//	strength_min = min(strength1, strength_min);
				//cout << strength1 << endl;
				//strength1 = sqrt(strength1);
			//	strength2 = sqrt(strength2);
				//strength3 = sqrt(strength3);
				//cout << strength1 << endl;

			if ((a <= b && b <= c) || (b <= c && c <= a) || (c <= a && a <= b))
			{
				//sum = -sum;
				//descriptor[i] = -descriptor[i];
				//descriptor[i+1] = -descriptor[i+1];
				//descriptor[i+2] = -descriptor[i+2];
				//descriptor[idx++] = 1;
				strength1 = -strength1;
				strength2 = -strength2;
				strength3 = -strength3;
			}
			else
			{
				// descriptor[idx++] = 0;
			}
			if (length == d * d * (n + 1))
			{
				descriptor[idx++] = strength1;
			}
		}
		/*
		for (idx = d * d * n; idx < d * d * (n + 1); idx++)
		{
			float temp = abs(descriptor[idx]);
			temp = (temp - strength_min) / (strength_max - strength_min);
			if (descriptor[idx] < 0)
			{
				temp = -temp;
			}
			descriptor[idx] = temp;
		}
		*/


	}

	/************************创建高斯金字塔第一组，第一层图像************************************/
	/*image表示输入原始图像
	 init_image表示生成的高斯尺度空间的第一层图像
	 */
	void MySift::create_initial_image(const Mat& image, Mat& init_image) const
	{
		//转换换为灰度图像
		Mat gray_image;
		if (image.channels() != 1)
			cvtColor(image, gray_image, CV_RGB2GRAY);
		else
			gray_image = image.clone();

		//转换到0-1之间的浮点类型数据，方便接下来的处理
		Mat floatImage;
		//float_image=(float)gray_image*(1.0/255.0)
		gray_image.convertTo(floatImage, CV_32FC1, 1.0 / 255.0, 0);
		double sig_diff = 0;
		if (double_size) {
			Mat temp_image;
			resize(floatImage, temp_image, Size(2 * floatImage.cols, 2 * floatImage.rows), 0, 0, INTER_LINEAR);
			sig_diff = sqrt(sigma * sigma - 4.0 * INIT_SIGMA * INIT_SIGMA);
			//高斯滤波窗口大小选择很重要，这里选择(4*sig_diff_1+1)-(6*sig_diff+1)之间
			int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig_diff) + 1;
			Size kernel_size(kernel_width, kernel_width);
			GaussianBlur(temp_image, init_image, kernel_size, sig_diff, sig_diff);
		}
		else {
			sig_diff = sqrt(sigma * sigma - 1.0 * INIT_SIGMA * INIT_SIGMA);
			//高斯滤波窗口大小选择很重要，这里选择(4*sig_diff_1+1)-(6*sig_diff+1)之间
			int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig_diff) + 1;
			Size kernel_size(kernel_width, kernel_width);
			GaussianBlur(floatImage, init_image, kernel_size, sig_diff, sig_diff);
		}
	}

	 /**************************生成高斯金字塔第二种方法******************************************/
	 /*init_image表示已经生成的高斯金字塔第一层图像
	  gauss_pyramid表示生成的高斯金字塔
	  nOctaves表示高斯金字塔的组数
	 */
	void MySift::build_gaussian_pyramid(const Mat& init_image, vector<vector<Mat>>& gauss_pyramid, int nOctaves) const
	{
		vector<double> sig;
		sig.push_back(sigma);
		double k = pow(2.0, 1.0 / nOctaveLayers);
		for (int i = 1; i < nOctaveLayers + 3; ++i) {
			double prev_sig = pow(k, double(i - 1)) * sigma;
			double curr_sig = k * prev_sig;
			sig.push_back(sqrt(curr_sig * curr_sig - prev_sig * prev_sig));
		}

		gauss_pyramid.resize(nOctaves);
		for (int i = 0; i < nOctaves; ++i)
		{
			gauss_pyramid[i].resize(nOctaveLayers + 3);
		}

		for (int i = 0; i < nOctaves; ++i)//对于每一组
		{
			for (int j = 0; j < nOctaveLayers + 3; ++j)//对于组内的每一层
			{
				if (i == 0 && j == 0)//第一组，第一层
					gauss_pyramid[0][0] = init_image;
				else if (j == 0)
				{
					resize(gauss_pyramid[i - 1][3], gauss_pyramid[i][0],
						Size(gauss_pyramid[i - 1][3].cols / 2,
							gauss_pyramid[i - 1][3].rows / 2), 0, 0, INTER_LINEAR);
				}
				else
				{
					//高斯滤波窗口大小选择很重要，这里选择(4*sig_diff_1+1)-(6*sig_diff+1)之间
					int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig[j]) + 1;
					Size kernel_size(kernel_width, kernel_width);
					GaussianBlur(gauss_pyramid[i][j - 1], gauss_pyramid[i][j], kernel_size, sig[j], sig[j]);
				}
			}
		}
	}

	/*******************生成高斯差分金字塔，即LOG金字塔*************************/
	/*dog_pyramid表示DOG金字塔
	 gauss_pyramin表示高斯金字塔*/
	void MySift::build_dog_pyramid(vector<vector<Mat>>& dog_pyramid, const vector<vector<Mat>>& gauss_pyramid) const
	{
		vector<vector<Mat>>::size_type nOctaves = gauss_pyramid.size();
		for (vector<vector<Mat>>::size_type i = 0; i < nOctaves; ++i)
		{
			vector<Mat> temp_vec;
			for (auto j = 0; j < nOctaveLayers + 2; ++j)
			{
				Mat temp_img = gauss_pyramid[i][j + 1] - gauss_pyramid[i][j];
				temp_vec.push_back(temp_img);
			}
			dog_pyramid.push_back(temp_vec);
			temp_vec.clear();
		}
	}


	/***********************该函数计算尺度空间特征点的主方向***************************/
	/*image表示特征点所在位置的高斯图像
	 pt表示特征点的位置坐标(x,y)
	 scale特征点的尺度
	 n表示直方图bin个数
	 hist表示计算得到的直方图
	 函数返回值是直方图hist中的最大数值*/
	static float clac_orientation_hist(const Mat& image, Point pt, float scale, int n, float* hist)
	{
		int radius = cvRound(ORI_RADIUS * scale);//特征点邻域半径(3*1.5*scale)
		int len = (2 * radius + 1) * (2 * radius + 1);//特征点邻域像素总个数（最大值）

		float sigma = ORI_SIG_FCTR * scale;//特征点邻域高斯权重标准差(1.5*scale)
		float exp_scale = -1.f / (2 * sigma * sigma);

		//使用AutoBuffer分配一段内存，这里多出4个空间的目的是为了方便后面平滑直方图的需要
		AutoBuffer<float> buffer(4 * len + n + 4);
		//X保存水平差分，Y保存数值差分，Mag保存梯度幅度，Ori保存梯度角度，W保存高斯权重
		float* X = buffer, * Y = buffer + len, * Mag = Y, * Ori = Y + len, * W = Ori + len;
		float* temp_hist = W + len + 2;//临时保存直方图数据

		for (int i = 0; i < n; ++i)
			temp_hist[i] = 0.f;//数据清零

		//计算邻域像素的水平差分和竖直差分
		int k = 0;
		for (int i = -radius; i < radius; ++i)
		{
			int y = pt.y + i;
			if (y <= 0 || y >= image.rows - 1)
				continue;
			for (int j = -radius; j < radius; ++j)
			{
				int x = pt.x + j;
				if (x <= 0 || x >= image.cols - 1)
					continue;

				float dx = image.at<float>(y, x + 1) - image.at<float>(y, x - 1);
				float dy = image.at<float>(y + 1, x) - image.at<float>(y - 1, x);
				X[k] = dx; Y[k] = dy; W[k] = (i * i + j * j) * exp_scale;
				++k;
			}
		}

		len = k;
		//计算邻域像素的梯度幅度,梯度方向，高斯权重
		cv::hal::exp(W, W, len);
		cv::hal::fastAtan2(Y, X, Ori, len, true);//角度范围0-360度
		cv::hal::magnitude(X, Y, Mag, len);

		for (int i = 0; i < len; ++i)
		{
			int bin = cvRound((n / 360.f) * Ori[i]);//bin的范围约束在[0,(n-1)]
			if (bin >= n)
				bin = bin - n;
			if (bin < 0)
				bin = bin + n;
			temp_hist[bin] = temp_hist[bin] + Mag[i] * W[i];
		}

		//平滑直方图
		temp_hist[-1] = temp_hist[n - 1];
		temp_hist[-2] = temp_hist[n - 2];
		temp_hist[n] = temp_hist[0];
		temp_hist[n + 1] = temp_hist[1];
		for (int i = 0; i < n; ++i)
		{
			hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * (1.f / 16.f) +
				(temp_hist[i - 1] + temp_hist[i + 1]) * (4.f / 16.f) +
				temp_hist[i] * (6.f / 16.f);
		}

		//获得直方图中最大值
		float max_value = hist[0];
		for (int i = 1; i < n; ++i)
		{
			if (hist[i] > max_value)
				max_value = hist[i];
		}
		return max_value;
	}

	/****************************该函数精确定位特征点位置(x,y,scale)*************************/
	/*dog_pry表示DOG金字塔
	 kpt表示精确定位后该特征点的信息
	 octave表示初始特征点所在的组
	 layer表示初始特征点所在的层
	 row表示初始特征点在图像中的行坐标
	 col表示初始特征点在图像中的列坐标
	 nOctaveLayers表示DOG金字塔每组中间层数，默认是3
	 contrastThreshold表示对比度阈值，默认是0.04
	 edgeThreshold表示边缘阈值，默认是10
	 sigma表示高斯尺度空间最底层图像尺度，默认是1.6*/
	static bool adjust_local_extrema(const vector<vector<Mat>>& dog_pyr, KeyPoint& kpt, int octave, int& layer,
		int& row, int& col, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma)
	{
		float xi = 0, xr = 0, xc = 0;
		int i = 0;
		for (; i < MAX_INTERP_STEPS; ++i)//最大迭代次数
		{
			const Mat& img = dog_pyr[octave][layer];//当前层图像索引
			const Mat& prev = dog_pyr[octave][layer - 1];//之前层图像索引
			const Mat& next = dog_pyr[octave][layer + 1];//下一层图像索引

			//特征点位置x方向，y方向,尺度方向的一阶偏导数
			float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * (1.f / 2.f);
			float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * (1.f / 2.f);
			float dz = (next.at<float>(row, col) - prev.at<float>(row, col)) * (1.f / 2.f);

			//计算特征点位置二阶偏导数
			float v2 = img.at<float>(row, col);
			float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
			float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
			float dzz = prev.at<float>(row, col) + next.at<float>(row, col) - 2 * v2;

			//计算特征点周围混合二阶偏导数
			float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
				img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) * (1.f / 4.f);
			float dxz = (next.at<float>(row, col + 1) + prev.at<float>(row, col - 1) -
				next.at<float>(row, col - 1) - prev.at<float>(row, col + 1)) * (1.f / 4.f);
			float dyz = (next.at<float>(row + 1, col) + prev.at<float>(row - 1, col) -
				next.at<float>(row - 1, col) - prev.at<float>(row + 1, col)) * (1.f / 4.f);

			Matx33f H(dxx, dxy, dxz,
				dxy, dyy, dyz,
				dxz, dyz, dzz);

			Vec3f dD(dx, dy, dz);

			Vec3f X = H.solve(dD, DECOMP_SVD);

			xc = -X[0];//x方向偏移量
			xr = -X[1];//y方向偏移量
			xi = -X[2];//尺度方向偏移量

			//如果三个方向偏移量都小于0.5，说明已经找到特征点准确位置
			if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xi) < 0.5f)
				break;

			//如果其中一个方向的偏移量过大，则删除该点
			if (abs(xc) > (float)(INT_MAX / 3) ||
				abs(xr) > (float)(INT_MAX / 3) ||
				abs(xi) > (float)(INT_MAX / 3))
				return false;

			col = col + cvRound(xc);
			row = row + cvRound(xr);
			layer = layer + cvRound(xi);

			//如果特征点定位在边界区域，同样也需要删除
			if (layer<1 || layer>nOctaveLayers ||
				col<BORDER || col>img.cols - BORDER ||
				row<BORDER || row>img.rows - BORDER)
				return false;
		}

		//如果i=MAX_INTERP_STEPS，说明循环结束也没有满足条件，即该特征点需要被删除
		if (i >= MAX_INTERP_STEPS)
			return false;

		/**************************再次删除低响应点********************************/
		//再次计算特征点位置x方向，y方向,尺度方向的一阶偏导数
		{
			const Mat& img = dog_pyr[octave][layer];
			const Mat& prev = dog_pyr[octave][layer - 1];
			const Mat& next = dog_pyr[octave][layer + 1];

			float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * (1.f / 2.f);
			float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * (1.f / 2.f);
			float dz = (next.at<float>(row, col) - prev.at<float>(row, col)) * (1.f / 2.f);
			Matx31f dD(dx, dy, dz);
			float t = dD.dot(Matx31f(xc, xr, xi));

			float contr = img.at<float>(row, col) + t * 0.5f;//特征点响应
			//Low建议contr阈值是0.03，但是RobHess等建议阈值为0.04/nOctaveLayers
			if (abs(contr) < contrastThreshold / nOctaveLayers)
				return false;


			/******************************删除边缘响应比较强的点************************************/
			//再次计算特征点位置二阶偏导数
			float v2 = img.at<float>(row, col);
			float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
			float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
			float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
				img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) * (1.f / 4.f);
			float det = dxx * dyy - dxy * dxy;
			float trace = dxx + dyy;
			if (det < 0 || (trace * trace * edgeThreshold >= det * (edgeThreshold + 1) * (edgeThreshold + 1)))
				return false;

			/*********到目前为止该特征的满足上面所有要求，保存该特征点信息***********/
			kpt.pt.x = ((float)col + xc) * (1 << octave);//相对于最底层的图像的x坐标
			kpt.pt.y = ((float)row + xr) * (1 << octave);//相对于最底层图像的y坐标
			kpt.octave = octave + (layer << 8);//组号保存在低字节，层号保存在高字节
			//相对于最底层图像的尺度
			kpt.size = sigma * powf(2.f, (layer + xi) / nOctaveLayers) * (1 << octave);
			kpt.response = abs(contr);//特征点响应值

			return true;
		}

	}


	/************该函数在DOG金字塔上进行特征点检测，特征点精确定位，删除低对比度点，删除边缘响应较大点**********/
	/*dog_pyr表示高斯金字塔
	 gauss_pyr表示高斯金字塔
	 keypoints表示检测到的特征点*/
	void MySift::find_scale_space_extrema(const vector<vector<Mat>>& dog_pyr, const vector<vector<Mat>>& gauss_pyr,
		vector<KeyPoint>& keypoints) const
	{
		int nOctaves = (int)dog_pyr.size();
		//Low文章建议threshold是0.03，Rob Hess等人使用0.04/nOctaveLayers作为阈值
		float threshold = (float)(contrastThreshold / nOctaveLayers);
		const int n = ORI_HIST_BINS;//n=36
		float hist[n];
		KeyPoint kpt;

		keypoints.clear();//先清空keypoints
		//int numKeys = 0;

		for (int i = 0; i < nOctaves; ++i)//对于每一组
		{
			for (int j = 1; j <= nOctaveLayers; ++j)//对于组内每一层
			{
				const Mat& curr_img = dog_pyr[i][j];//当前层
				const Mat& prev_img = dog_pyr[i][j - 1];//之前层
				const Mat& next_img = dog_pyr[i][j + 1];
				int num_row = curr_img.rows;
				int num_col = curr_img.cols;//获得当前组图像的大小
				size_t step = curr_img.step1();//一行元素所占宽度

				for (int r = BORDER; r < num_row - BORDER; ++r)
				{
					const float* curr_ptr = curr_img.ptr<float>(r);
					const float* prev_ptr = prev_img.ptr<float>(r);
					const float* next_ptr = next_img.ptr<float>(r);

					for (int c = BORDER; c < num_col - BORDER; ++c)
					{
						float val = curr_ptr[c];//当前中心点响应值

						//开始检测特征点
						if (abs(val) > threshold &&
							((val > 0 && val >= curr_ptr[c - 1] && val >= curr_ptr[c + 1] &&
								val >= curr_ptr[c - step - 1] && val >= curr_ptr[c - step] && val >= curr_ptr[c - step + 1] &&
								val >= curr_ptr[c + step - 1] && val >= curr_ptr[c + step] && val >= curr_ptr[c + step + 1] &&
								val >= prev_ptr[c] && val >= prev_ptr[c - 1] && val >= prev_ptr[c + 1] &&
								val >= prev_ptr[c - step - 1] && val >= prev_ptr[c - step] && val >= prev_ptr[c - step + 1] &&
								val >= prev_ptr[c + step - 1] && val >= prev_ptr[c + step] && val >= prev_ptr[c + step + 1] &&
								val >= next_ptr[c] && val >= next_ptr[c - 1] && val >= next_ptr[c + 1] &&
								val >= next_ptr[c - step - 1] && val >= next_ptr[c - step] && val >= next_ptr[c - step + 1] &&
								val >= next_ptr[c + step - 1] && val >= next_ptr[c + step] && val >= next_ptr[c + step + 1]) ||
								(val < 0 && val <= curr_ptr[c - 1] && val <= curr_ptr[c + 1] &&
									val <= curr_ptr[c - step - 1] && val <= curr_ptr[c - step] && val <= curr_ptr[c - step + 1] &&
									val <= curr_ptr[c + step - 1] && val <= curr_ptr[c + step] && val <= curr_ptr[c + step + 1] &&
									val <= prev_ptr[c] && val <= prev_ptr[c - 1] && val <= prev_ptr[c + 1] &&
									val <= prev_ptr[c - step - 1] && val <= prev_ptr[c - step] && val <= prev_ptr[c - step + 1] &&
									val <= prev_ptr[c + step - 1] && val <= prev_ptr[c + step] && val <= prev_ptr[c + step + 1] &&
									val <= next_ptr[c] && val <= next_ptr[c - 1] && val <= next_ptr[c + 1] &&
									val <= next_ptr[c - step - 1] && val <= next_ptr[c - step] && val <= next_ptr[c - step + 1] &&
									val <= next_ptr[c + step - 1] && val <= next_ptr[c + step] && val <= next_ptr[c + step + 1])))
						{
							//++numKeys;
							//获得特征点初始行号，列号，组号，组内层号
							int r1 = r, c1 = c, octave = i, layer = j;
							if (!adjust_local_extrema(dog_pyr, kpt, octave, layer, r1, c1,
								nOctaveLayers, (float)contrastThreshold,
								(float)edgeThreshold, (float)sigma))
							{
								continue;//如果该初始点不满足条件，则不保存改点
							}

							float scale = kpt.size / float(1 << octave);//该特征点相对于本组的尺度
							float max_hist = clac_orientation_hist(gauss_pyr[octave][layer],
								Point(c1, r1), scale, n, hist);
							float mag_thr = max_hist * ORI_PEAK_RATIO;

							for (int i = 0; i < n; ++i)
							{
								int left = 0, right = 0;
								if (i == 0)
									left = n - 1;
								else
									left = i - 1;

								if (i == n - 1)
									right = 0;
								else
									right = i + 1;

								if (hist[i] > hist[left] && hist[i] > hist[right] && hist[i] >= mag_thr)
								{
									float bin = i + 0.5f * (hist[left] - hist[right]) / (hist[left] + hist[right] - 2 * hist[i]);
									if (bin < 0)
										bin = bin + n;
									if (bin >= n)
										bin = bin - n;

									kpt.angle = (360.f / n) * bin;//特征点的主方向0-360度
									keypoints.push_back(kpt);//保存该特征点

								}

							}
						}
					}
				}
			}
		}

		//cout << "初始满足要求特征点个数是: " << numKeys << endl;
	}


	/******************************计算一个特征点描的述子***********************************/
	/*gauss_image表示特征点所在的高斯图像
	 main_angle表示特征点的主方向，角度范围是0-360度
	 pt表示特征点在高斯图像上的坐标，相对与本组，不是相对于最底层
	 scale表示特征点所在层的尺度，相对于本组，不是相对于最底层
	 d表示特征点邻域网格宽度
	 n表示每个网格内像素梯度角度等分个数
	 descriptor表示生成的特征点的描述子*/
	static void calc_sift_descriptor(const Mat& gauss_image, float main_angle, Point2f pt,
		float scale, int d, int n, float* descriptor)
	{
		Point ptxy(cvRound(pt.x), cvRound(pt.y));//坐标取整
		float cos_t = cosf(-main_angle * (float)(CV_PI / 180));
		float sin_t = sinf(-main_angle * (float)(CV_PI / 180));
		float bins_per_rad = n / 360.f;//n=8
		float exp_scale = -1.f / (d * d * 0.5f);
		float hist_width = DESCR_SCL_FCTR * scale;//每个网格的宽度
		int radius = cvRound(hist_width * (d + 1) * sqrt(2) * 0.5f);//特征点邻域半径

		int rows = gauss_image.rows, cols = gauss_image.cols;
		radius = min(radius, (int)sqrt((double)rows * rows + cols * cols));
		cos_t = cos_t / hist_width;
		sin_t = sin_t / hist_width;

		int len = (2 * radius + 1) * (2 * radius + 1);
		int histlen = (d + 2) * (d + 2) * (n + 2);

		AutoBuffer<float> buf(6 * len + histlen);
		//X保存水平差分，Y保存竖直差分，Mag保存梯度幅度，Angle保存特征点方向,W保存高斯权重
		float* X = buf, * Y = buf + len, * Mag = Y, * Angle = Y + len, * W = Angle + len;
		float* RBin = W + len, * CBin = RBin + len, * hist = CBin + len;

		//首先清空直方图数据
		for (int i = 0; i < d + 2; ++i)
		{
			for (int j = 0; j < d + 2; ++j)
			{
				for (int k = 0; k < n + 2; ++k)
					hist[(i * (d + 2) + j) * (n + 2) + k] = 0.f;
			}
		}

		//计算特征点邻域范围内每个像素的差分核高斯权重的指数部分
		int k = 0;
		for (int i = -radius; i < radius; ++i)
		{
			for (int j = -radius; j < radius; ++j)
			{
				float c_rot = j * cos_t - i * sin_t;
				float r_rot = j * sin_t + i * cos_t;
				float rbin = r_rot + d / 2 - 0.5f;
				float cbin = c_rot + d / 2 - 0.5f;
				int r = ptxy.y + i, c = ptxy.x + j;

				//这里rbin,cbin范围是(-1,d)
				if (rbin > -1 && rbin<d && cbin>-1 && cbin < d &&
					r>0 && r < rows - 1 && c>0 && c < cols - 1)
				{
					float dx = gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
					float dy = gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);
					X[k] = dx; //水平差分
					Y[k] = dy;//竖直差分
					RBin[k] = rbin;
					CBin[k] = cbin;
					W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;//高斯权值的指数部分
					++k;
				}
			}
		}

		//计算像素梯度幅度，梯度角度，和高斯权值
		len = k;
		cv::hal::fastAtan2(Y, X, Angle, len, true);//角度范围是0-360度
		cv::hal::magnitude(X, Y, Mag, len);//幅度
		cv::hal::exp(W, W, len);//高斯权值

		//计算每个特征点的描述子
		for (k = 0; k < len; ++k)
		{
			float rbin = RBin[k], cbin = CBin[k];//rbin,cbin范围是(-1,d)
			float obin = (Angle[k] - main_angle) * bins_per_rad;
			float mag = Mag[k] * W[k];

			int r0 = cvFloor(rbin);//ro取值集合是{-1,0,1,2，3}
			int c0 = cvFloor(cbin);//c0取值集合是{-1，0，1，2，3}
			int o0 = cvFloor(obin);
			rbin = rbin - r0;
			cbin = cbin - c0;
			obin = obin - o0;

			//限制范围为[0,n)
			if (o0 < 0)
				o0 = o0 + n;
			if (o0 >= n)
				o0 = o0 - n;

			//使用三线性插值方法，计算直方图
			float v_r1 = mag * rbin;//第二行分配的值
			float v_r0 = mag - v_r1;//第一行分配的值

			float v_rc11 = v_r1 * cbin;
			float v_rc10 = v_r1 - v_rc11;
			float v_rc01 = v_r0 * cbin;
			float v_rc00 = v_r0 - v_rc01;

			float v_rco111 = v_rc11 * obin;
			float v_rco110 = v_rc11 - v_rco111;

			float v_rco101 = v_rc10 * obin;
			float v_rco100 = v_rc10 - v_rco101;

			float v_rco011 = v_rc01 * obin;
			float v_rco010 = v_rc01 - v_rco011;

			float v_rco001 = v_rc00 * obin;
			float v_rco000 = v_rc00 - v_rco001;

			//该像素所在网格的索引
			int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
			hist[idx] += v_rco000;
			hist[idx + 1] += v_rco001;
			hist[idx + n + 2] += v_rco010;
			hist[idx + n + 3] += v_rco011;
			hist[idx + (d + 2) * (n + 2)] += v_rco100;
			hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
			hist[idx + (d + 3) * (n + 2)] += v_rco110;
			hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
		}

		//由于圆周循环的特性，对计算以后幅角小于 0 度或大于 360 度的值重新进行调整，使
		//其在 0～360 度之间
		for (int i = 0; i < d; ++i)
		{
			for (int j = 0; j < d; ++j)
			{
				int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
				hist[idx] += hist[idx + n];
				//hist[idx + 1] += hist[idx + n + 1];//opencv源码中这句话是多余的,hist[idx + n + 1]永远是0.0
				for (k = 0; k < n; ++k)
					descriptor[(i * d + j) * n + k] = hist[idx + k];
			}
		}

		//对描述子进行归一化
		int lenght = d * d * n;
		float norm = 0;
		for (int i = 0; i < lenght; ++i)
		{
			norm = norm + descriptor[i] * descriptor[i];
		}
		norm = sqrt(norm);
		for (int i = 0; i < lenght; ++i)
		{
			descriptor[i] = descriptor[i] / norm;
		}

		//阈值截断
		for (int i = 0; i < lenght; ++i)
		{
			descriptor[i] = min(descriptor[i], DESCR_MAG_THR);
		}

		//再次归一化
		norm = 0;
		for (int i = 0; i < lenght; ++i)
		{
			norm = norm + descriptor[i] * descriptor[i];
		}
		norm = sqrt(norm);
		for (int i = 0; i < lenght; ++i)
		{
			descriptor[i] = descriptor[i] / norm;
		}

	}


	/********************************该函数计算所有特征点特征描述子***************************/
	/*gauss_pyr表示高斯金字塔
	 keypoints表示特征点、
	 descriptors表示生成的特征点的描述子*/
	void MySift::calc_descriptors(const vector<vector<Mat>>& gauss_pyr, vector<KeyPoint>& keypoints,
		Mat& descriptors) const
	{
		int d = DESCR_WIDTH;//d=4,特征点邻域网格个数是d x d
		int n = DESCR_HIST_BINS;//n=8,每个网格特征点梯度角度等分为8个方向
		int length = d * d * (n + 1);
		descriptors.create(keypoints.size(), length, CV_32FC1);//分配空间
		vector<vector<float> > degree;
		degree = get_degree();
		for (size_t i = 0; i < keypoints.size(); ++i)//对于每一个特征点
		{
			int octaves, layer;
			//得到特征点所在的组号，层号
			octaves = keypoints[i].octave & 255;
			layer = (keypoints[i].octave >> 8) & 255;

			//得到特征点相对于本组的坐标，不是最底层
			Point2f pt(keypoints[i].pt.x / (1 << octaves), keypoints[i].pt.y / (1 << octaves));
			float scale = keypoints[i].size / (1 << octaves);//得到特征点相对于本组的尺度
			float main_angle = keypoints[i].angle;//特征点主方向

			//计算改点的描述子
			//calc_sift_descriptor(gauss_pyr[octaves][layer],main_angle, pt, scale,d, n, descriptors.ptr<float>((int)i));
			//new2_calc_sift_descriptor(gauss_pyr[octaves][layer], main_angle, pt, scale, d, n, descriptors.ptr<float>((int)i), degree, length);
			new_calc_sift_descriptor(gauss_pyr[octaves][layer], main_angle, pt, scale, d, n, descriptors.ptr<float>((int)i), degree, length);
			if (double_size)//如果图像尺寸扩大一倍
			{
				keypoints[i].pt.x = keypoints[i].pt.x / 2.f;
				keypoints[i].pt.y = keypoints[i].pt.y / 2.f;
			}
		}

	}

	/******************************特征点检测*********************************/
	/*image表示输入的图像
	 gauss_pyr表示生成的高斯金字塔
	 dog_pyr表示生成的高斯差分DOG金字塔
	 keypoints表示检测到的特征点*/
	void MySift::detect(const Mat& image, vector<vector<Mat>>& gauss_pyr, vector<vector<Mat>>& dog_pyr,
		vector<KeyPoint>& keypoints) const
	{
		if (image.empty() || image.depth() != CV_8U)
			CV_Error(CV_StsBadArg, "输入图像为空，或者图像深度不是CV_8U");


		//计算高斯金字塔组数
		int nOctaves;
		nOctaves = num_octaves(image);

		//生成高斯金字塔第一层图像
		Mat init_gauss;
		create_initial_image(image, init_gauss);

		//生成高斯尺度空间图像
		build_gaussian_pyramid(init_gauss, gauss_pyr, nOctaves);

		//生成高斯差分金字塔(DOG金字塔，or LOG金字塔)
		build_dog_pyramid(dog_pyr, gauss_pyr);

		//在DOG金字塔上检测特征点
		find_scale_space_extrema(dog_pyr, gauss_pyr, keypoints);

		//如果指定了特征点个数,并且设定的设置小于默认检测的特征点个数
		if (nfeatures != 0 && nfeatures < (int)keypoints.size())
		{
			//特征点响应值从大到小排序
			sort(keypoints.begin(), keypoints.end(),
				[](const KeyPoint& a, const KeyPoint& b)
				{return a.response > b.response; });

			//删除点多余的特征点
			keypoints.erase(keypoints.begin() + nfeatures, keypoints.end());
		}


	}

	/**********************特征点描述*******************/
	/*gauss_pyr表示高斯金字塔
	 keypoints表示特征点集合
	 descriptors表示特征点的描述子*/
	void MySift::comput_des(const vector<vector<Mat>>& gauss_pyr, vector<KeyPoint>& keypoints, Mat& descriptors) const
	{
		calc_descriptors(gauss_pyr, keypoints, descriptors);
	}
}
	
