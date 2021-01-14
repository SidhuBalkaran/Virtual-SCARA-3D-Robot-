////////////////////////////////////////////////////////////////
// ELEX 7825 Template project for BCIT
// Created Sept 9, 2020 by Craig Hennessey
// Last updated Nov 8, 2020
////////////////////////////////////////////////////////////////
#include "stdafx.h"

#include "Robot.h"

#include<stdio.h>
#include <string>
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <cmath>
#include <fstream>
#include <opencv2/aruco/charuco.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace dnn;
using namespace aruco;
using namespace cv;
using namespace std;

int menu() {
    int choice;
    cout << "\nPrint Menu\n";
    cout << "1) Manual\n";
    cout << "2) Fixed Joints FWD kin\n";
    cout << "3) Automatic\n";
    cout << "4) inverse kinematics\n";
    cout << "5) Inverse KInematic line track\n";
    cout << "6) Exit\n";
    cout << "7) ctraj\n";
    cout << "8) jtraj\n";
    cout << "9) Detect Board\n";
    cout << "10) Calibrate Board\n";
    cout << "11) YOLO\n";
    cin >> choice;
    return choice;
}

int main(int argc, char* argv[])
{
    Robot robot;
    Size image_size = Size(1000, 600);
    //image matrix to draw everything on
    cv::Mat im = cv::Mat::zeros(image_size, CV_8UC3) + CV_RGB(60, 60, 60);

    while (1) {
        //the operation mode selected
        int mode = menu();
        if (mode == 1) {
            float J1, J2, J3, J4;
            cout << "J1:";
            cin >> J1;
            cout << "J2:";
            cin >> J2;
            cout << "J3:";
            cin >> J3;
            cout << "J4:";
            cin >> J4;
            cout << robot.fkine(J1, J2, J3, J4);
            robot.manualDraw(im, J1, J2, J3, J4);
            cv::waitKey(0);
        }
        if (mode == 2) {
            cout << robot.fkine(20, 30, 45, 40);
            robot.manualDraw(im, 20, 30, 45, 40);
            cv::waitKey(0);
        }
        if (mode == 3) {
            cv::Mat im = cv::Mat::zeros(image_size, CV_8UC3) + CV_RGB(60, 60, 60);
            robot.animate(im);
        }
        if (mode == 4) {
            vector<float> q;
            float q1, q2;
            int choice;
            float x, y, z, angle;
            cout << "X:";
            cin >> x;
            cout << "Y:";
            cin >> y;
            cout << "Z:";
            cin >> z;
            cout << "Angle:";
            cin >> angle;

            q = robot.ikine(x, y, angle, z);

            cout << "Choose solution (1) and (2)\n";
            cin >> choice;
            if (choice == 1) {
                q1 = q.at(0);
                q2 = q.at(1);
            }
            else {
                q1 = q.at(2);
                q2 = q.at(3);
            }
            robot.manualDraw(im, q1 * (180 / CV_PI), q2 * (180 / CV_PI), angle, -z);
            cv::waitKey(0);
        }
        if (mode == 5) {
            vector<float> q;
            for (float x = -100; x <= 100; x++) {

                cv::Mat im = cv::Mat::zeros(image_size, CV_8UC3) + CV_RGB(60, 60, 60);

                q = robot.ikine(x, 50, -50, 0);
                robot.manualDraw(im, q.at(2) * (180 / CV_PI), q.at(3) * (180 / CV_PI), 0, -(-50));
                cv::waitKey(33);
                if (x == 100)
                    cv::waitKey(0);
            }
        }
        if (mode == 7)
        {
            int timeSteps = 200;
            Mat t = Mat::zeros(cv::Size(1, timeSteps), CV_32FC1);
            Mat initPos = Mat::zeros(cv::Size(1, 2), CV_32FC1);
            Mat initVel = Mat::zeros(cv::Size(1, 2), CV_32FC1);
            Mat finPos = Mat::zeros(cv::Size(1, 2), CV_32FC1);
            Mat finVel = Mat::zeros(cv::Size(1, 2), CV_32FC1);

            initPos.at<float>(0, 0) = 200;
            initPos.at<float>(1, 0) = 100;

            finPos.at<float>(0, 0) = 150;
            finPos.at<float>(1, 0) = 150;

            initVel.at<float>(0, 0) = 0;
            initVel.at<float>(1, 0) = 0;

            finVel.at<float>(0, 0) = 5;
            finVel.at<float>(1, 0) = 5;

            for (int i = 0; i < timeSteps; i++) {
                t.at<float>(i, 0) = i;
            }

            robot.ctraj(im, initPos, finPos, initVel, finVel, t);

            finPos.at<float>(0, 0) = -100;
            finPos.at<float>(1, 0) = 200;

            initPos.at<float>(0, 0) = 150;
            initPos.at<float>(1, 0) = 150;

            robot.ctraj(im, initPos, finPos, initVel, finVel, t);

            initPos.at<float>(0, 0) = -100;
            initPos.at<float>(1, 0) = 200;

            finPos.at<float>(0, 0) = 200;
            finPos.at<float>(1, 0) = 200;

            robot.ctraj(im, initPos, finPos, initVel, finVel, t);
        }
        if (mode == 8) {
            int timeSteps = 50;
            Mat t = Mat::zeros(cv::Size(1, timeSteps), CV_32FC1);
            Mat initPos = Mat::zeros(cv::Size(1, 2), CV_32FC1);
            Mat initVel = Mat::zeros(cv::Size(1, 2), CV_32FC1);
            Mat finPos = Mat::zeros(cv::Size(1, 2), CV_32FC1);
            Mat finVel = Mat::zeros(cv::Size(1, 2), CV_32FC1);

            initPos.at<float>(0, 0) = 0;
            initPos.at<float>(1, 0) = 90;

            finPos.at<float>(0, 0) = 0;
            finPos.at<float>(1, 0) = 0;

            initVel.at<float>(0, 0) = 0;
            initVel.at<float>(1, 0) = 0;

            finVel.at<float>(0, 0) = 5;
            finVel.at<float>(1, 0) = 5;

            for (int i = 0; i < timeSteps; i++) {
                t.at<float>(i, 0) = i;
            }

            Mat J1 = robot.jtraj(im, initPos, finPos, initVel, finVel, t);

            initPos.at<float>(0, 0) = 0;
            initPos.at<float>(1, 0) = 90;

            finPos.at<float>(0, 0) = 0;
            finPos.at<float>(1, 0) = 0;

            Mat J2 = robot.jtraj(im, initPos, finPos, initVel, finVel, t);

            initPos.at<float>(0, 0) = 0;
            initPos.at<float>(1, 0) = 90;

            finPos.at<float>(0, 0) = 0;
            finPos.at<float>(1, 0) = 0;

            Mat J3 = robot.jtraj(im, initPos, finPos, initVel, finVel, t);

            initPos.at<float>(0, 0) = 0;
            initPos.at<float>(1, 0) = 100;

            finPos.at<float>(0, 0) = 0;
            finPos.at<float>(1, 0) = 0;

            Mat J4 = robot.jtraj(im, initPos, finPos, initVel, finVel, t);

            for (int i = 0; i < timeSteps; i++) {
                robot.manualDraw(im, J1.at<float>(i, 1), J2.at<float>(i, 1), J3.at<float>(i, 1), J4.at<float>(i, 1));
                cv::waitKey(100);

            }

            initPos.at<float>(0, 0) = 0;
            initPos.at<float>(1, 0) = 0;

            finPos.at<float>(0, 0) = 0;
            finPos.at<float>(1, 0) = 90;


            for (int i = 0; i < timeSteps; i++) {
                t.at<float>(i, 0) = i;
            }

            J1 = robot.jtraj(im, initPos, finPos, initVel, finVel, t);

            initPos.at<float>(0, 0) = 0;
            initPos.at<float>(1, 0) = 0;

            finPos.at<float>(0, 0) = 0;
            finPos.at<float>(1, 0) = 90;

            J2 = robot.jtraj(im, initPos, finPos, initVel, finVel, t);

            initPos.at<float>(0, 0) = 0;
            initPos.at<float>(1, 0) = 0;

            finPos.at<float>(0, 0) = 0;
            finPos.at<float>(1, 0) = 90;

            J3 = robot.jtraj(im, initPos, finPos, initVel, finVel, t);

            initPos.at<float>(0, 0) = 0;
            initPos.at<float>(1, 0) = 0;

            finPos.at<float>(0, 0) = 0;
            finPos.at<float>(1, 0) = 100;

            J4 = robot.jtraj(im, initPos, finPos, initVel, finVel, t);

            for (int i = 0; i < timeSteps; i++) {
                robot.manualDraw(im, J1.at<float>(i, 1), J2.at<float>(i, 1), J3.at<float>(i, 1), J4.at<float>(i, 1));
                cv::waitKey(100);
            }
        }
        if (mode == 9) {
            robot.detect(im,argc,argv);
        }
        if (mode == 10) {
            robot.Calibrate(argc, argv);
        } 
        if (mode == 6) {
            break;
        }
        if (mode == 11) {
            robot.YOLO(im);
        }
    }
    return 1;
}

