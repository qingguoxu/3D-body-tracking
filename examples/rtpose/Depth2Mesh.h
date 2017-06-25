
#ifndef	__WHMIN_DEPTH2MESH_H__
#define __WHMIN_DEPTH2MESH_H__
#define _CRT_SECURE_NO_DEPRECATE
#include <windows.h>
//#include "../lib/PROJECTIVE_TET_MESH.h"
//#include "../lib/MESH.h"
#include "MESH.h"
#include "PROJECTIVE_TET_MESH.h"
#include <opencv.hpp>
#include <opencv/highgui.h>
#include <ctime>
#define	FLOAT_TYPE	float
using namespace std;
using namespace cv;

const Mat KK = (Mat_<float>(3, 3) << 365.5953, 0.0, 260.1922, 0.0, 365.5953, 209.5835, 0.0, 0.0, 1.0);
const Mat Rt = (Mat_<float>(3, 4) << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv,
	cv::Mat1i &X, cv::Mat1i &Y);

void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
	 cv::Mat1i &X, cv::Mat1i &Y);


void meshgridG(const cv::Mat &xgv, const cv::Mat &ygv,
	cv::Mat &X, cv::Mat &Y);

void meshgridTestG(const cv::Range &xgv, const cv::Range &ygv,
	cv::Mat &X, cv::Mat &Y);

Mat depthmap2pointmap(Mat& depth);
Mat MaskBlob(Mat& Mask, int num);

Mat im2bw(Mat src, double grayThresh);
Mat MaskBlob(Mat& mask);
void pointmap2mesh(Mat& pointmap, Mat& mask, MESH<FLOAT_TYPE>& submesh);
void selectmesh_vind(MESH<FLOAT_TYPE>& mesh, Mat& subv_index, int width, int height, MESH<FLOAT_TYPE>& submesh);
void RemoveLongFace(MESH<FLOAT_TYPE>& mesh, float threshold, MESH<FLOAT_TYPE>& mesh_clean);

#endif