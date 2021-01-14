#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <ctime>

#include "stdafx.h"
#include <string>
#include <iostream>
#include <thread>
#include <math.h>  
#include <vector>

// OpenGL
using namespace cv;
using namespace std;
using namespace dnn;


class Robot
{
public:
	Robot();
	~Robot();

	float view_angle = 0.2; // 0.2 looks good

	void manualDraw(Mat& im, float J1, float J2, float J3, float J4);

	void animate(Mat& im);

	Mat createHT(float tx, float ty, float tz, float rx, float ry, float rz);

	std::vector<Mat> createBox(float w, float h, float d);

	void transformBox(std::vector<Mat>& box, Mat T);

	void makeBox(Mat& im, std::vector<Mat> box, Scalar colour);

	void drawPose(Mat& im, Mat T);

	Mat fkine(float J1, float J2, float J3, float J4);

	vector<float> ikine(float x, float y, float J3, float J4);

	void drawUnitOr(Mat& im, Mat T);

	void ctraj(Mat& im, Mat initPos, Mat finPos, Mat initVel, Mat finVel, Mat timeSteps);

	Mat jtraj(Mat& im, Mat initPos, Mat finPos, Mat initVel, Mat finVel, Mat timeSteps);

	bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs);

    bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters>& params);

	int detect(Mat& im,int argc, char* argv[]);

	bool saveCameraParams(const string& filename, Size imageSize, float aspectRatio, int flags, const Mat& cameraMatrix, const Mat& distCoeffs, double totalAvgErr);

	int Robot::Calibrate(int argc, char* argv[]);

	void YOLO(Mat &im);
};