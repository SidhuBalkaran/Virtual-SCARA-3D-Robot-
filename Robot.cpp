#include "stdafx.h"

#include "Robot.h"
#include <cmath>

#include <opencv2/aruco/charuco.hpp>

namespace {
    const char* about = "Pose estimation using a ChArUco board";
    const char* keys =
        "{w        |       | Number of squares in X direction }"
        "{h        |       | Number of squares in Y direction }"
        "{sl       |       | Square side length (in meters) }"
        "{ml       |       | Marker side length (in meters) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{c        |       | Output file with calibrated camera parameters }"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{dp       |       | File of marker detector parameters }"
        "{rs       |       | Apply refind strategy }"
        "{r        |       | show rejected candidates too }";
}
namespace {

    const char* keys1 =
        "{w        |       | Number of squares in X direction }"
        "{h        |       | Number of squares in Y direction }"
        "{sl       |       | Square side length (in meters) }"
        "{ml       |       | Marker side length (in meters) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{@outfile |<none> | Output file with calibrated camera parameters }"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{dp       |       | File of marker detector parameters }"
        "{rs       | false | Apply refind strategy }"
        "{zt       | false | Assume zero tangential distortion }"
        "{a        |       | Fix aspect ratio (fx/fy) to this value }"
        "{pc       | false | Fix the principal point at the center }"
        "{sc       | false | Show detected chessboard corners after calibration }";
}

Robot::Robot()
{

}

Mat Robot::createHT(float tx, float ty, float tz, float rx, float ry, float rz)
{
    float r11, r12, r13, r21, r22, r23, r31, r32, r33;

    r11 = cos(rz) * cos(ry);
    r12 = cos(rz) * sin(ry) * sin(rx) - sin(rz) * cos(rx);
    r13 = cos(rz) * sin(ry) * cos(rx) + sin(rz) * sin(rx);
    r21 = sin(rz) * cos(ry);
    r22 = sin(rz) * sin(ry) * sin(rx) + cos(rz) * cos(rx);
    r23 = sin(rz) * sin(ry) * cos(rx) - cos(rz) * sin(rx);
    r31 = -1.0 * sin(ry);
    r32 = cos(ry) * sin(rx);
    r33 = cos(ry) * cos(rx);

    return (Mat1f(4, 4) << r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1);
}

std::vector<Mat> Robot::createBox(float w, float h, float d)
{
    std::vector <Mat> box;

    // The 8 vertexes, origin at the middle of the box
    box.push_back(Mat((Mat1f(4, 1) << -w / 2, -h / 2, -d / 2, 1)));
    box.push_back(Mat((Mat1f(4, 1) << w / 2, -h / 2, -d / 2, 1)));
    box.push_back(Mat((Mat1f(4, 1) << w / 2, h / 2, -d / 2, 1)));
    box.push_back(Mat((Mat1f(4, 1) << -w / 2, h / 2, -d / 2, 1)));
    box.push_back(Mat((Mat1f(4, 1) << -w / 2, -h / 2, d / 2, 1)));
    box.push_back(Mat((Mat1f(4, 1) << w / 2, -h / 2, d / 2, 1)));
    box.push_back(Mat((Mat1f(4, 1) << w / 2, h / 2, d / 2, 1)));
    box.push_back(Mat((Mat1f(4, 1) << -w / 2, h / 2, d / 2, 1)));

    // Move origin to middle of the the left hand face
    Mat T = createHT(w / 2, 0, 0, CV_PI, 0, 0);
    for (int i = 0; i < box.size(); i++)
    {
        box.at(i) = T * box.at(i);
    }
    return box;
}

void Robot::transformBox(std::vector<Mat>& box, Mat T)
{
    for (int i = 0; i < box.size(); i++)
    {
        box.at(i) = T * box.at(i);
    }
}

void Robot::makeBox(Mat& im, std::vector<Mat> box, Scalar colour)
{
    // The 12 lines connecting all vertexes 
    float draw_box1[] = { 0,1,2,3,4,5,6,7,0,1,2,3 };
    float draw_box2[] = { 1,2,3,0,5,6,7,4,4,5,6,7 };

    //lines between points
    for (int i = 0; i < 12; i++)
    {
        Point pt1 = Point2f(box.at(draw_box1[i]).at<float>(0, 0), box.at(draw_box1[i]).at<float>(1, 0));
        Point pt2 = Point2f(box.at(draw_box2[i]).at<float>(0, 0), box.at(draw_box2[i]).at<float>(1, 0));
        line(im, pt1, pt2, colour, 1);
    }

}

void Robot::drawPose(Mat& im, Mat T)
{


}

Mat Robot::fkine(float J1, float J2, float J3, float J4)
{
    J1 = J1 * (CV_PI / 180);
    J2 = J2 * (CV_PI / 180);
    J3 = J3 * (CV_PI / 180);
    Mat ET1 = createHT(0, 0, 0, 0, 0, J1);
    Mat ET2 = createHT(200, 0, 0, 0, 0, J2);
    Mat ET3 = createHT(200, 0, -J4, 0, 0, J3);
    return ET1 * ET2 * ET3;
}

std::vector<float> Robot::ikine(float x, float y, float J3, float J4)
{
    int a1, a2;
    a1 = a2 = 200;
    std::vector<float> q;

    float q1_1, q1_2, q2_1, q2_2, q3;

    q1_1 = 2 * atan2((400 * y + sqrt(-pow(x, 4) - 2 * pow(x, 2) * pow(y, 2) + 160000 * pow(x, 2) - pow(y, 4) + 160000 * pow(y, 2))), (pow(x, 2) + 400 * x + pow(y, 2)));
    q1_2 = 2 * atan2((400 * y - sqrt(-pow(x, 4) - 2 * pow(x, 2) * pow(y, 2) + 160000 * pow(x, 2) - pow(y, 4) + 160000 * pow(y, 2))), (pow(x, 2) + 400 * x + pow(y, 2)));

    q2_1 = -2 * atan2(sqrt(-pow(x, 2) - pow(y, 2) + 160000), sqrt(pow(x, 2) + pow(y, 2)));
    q2_2 = 2 * atan2(sqrt(-pow(x, 2) - pow(y, 2) + 160000), sqrt(pow(x, 2) + pow(y, 2)));

    q3 = -J3 * (CV_PI / 180) + q1_1 + q2_1;
    //cout << "q1: " << q1_1 * (180 / CV_PI) << "\n";
    //cout << "q2:" << q2_1 * (180 / CV_PI) << "\n";
    //cout << "q3:" << q3 * (180 / CV_PI) << "\n";
   // cout << "Position :" << J4 << '\n';

    q.push_back(q1_1);
    q.push_back(q2_1);
    q.push_back(q1_2);
    q.push_back(q2_2);

    return q;
}

void Robot::drawUnitOr(Mat& im, Mat T)
{
    Mat Org0 = (Mat1f(4, 1) << 0, 0, 0, 1);
    Mat Org_x = (Mat1f(4, 1) << 20, 0, 0, 1);
    Mat Org_y = (Mat1f(4, 1) << 0, 20, 0, 1);
    Mat Org_z = (Mat1f(4, 1) << 0, 0, 20, 1);

    Org0 = T * Org0;
    Org_x = T * Org_x;
    Org_y = T * Org_y;
    Org_z = T * Org_z;

    Point pt_1 = Point2f(Org0.at<float>(0, 0), Org0.at<float>(1, 0));
    Point pt_2 = Point2f(Org_x.at<float>(0, 0), Org_x.at<float>(1, 0));
    line(im, pt_1, pt_2, CV_RGB(255, 0, 0), 2);
    Point pt_3 = Point2f(Org0.at<float>(0, 0), Org0.at<float>(1, 0));
    Point pt_4 = Point2f(Org_y.at<float>(0, 0), Org_y.at<float>(1, 0));
    line(im, pt_3, pt_4, CV_RGB(0, 255, 0), 2);
    Point pt_5 = Point2f(Org0.at<float>(0, 0), Org0.at<float>(1, 0));
    Point pt_6 = Point2f(Org_z.at<float>(0, 0), Org_z.at<float>(1, 0));
    line(im, pt_5, pt_6, CV_RGB(0, 0, 255), 2);
}

void Robot::manualDraw(Mat& im, float J1, float J2, float J3, float J4)
{
    Size image_size = Size(1000, 600);
    im = cv::Mat::zeros(image_size, CV_8UC3) + CV_RGB(60, 60, 60);
    //creating 3 robot arms
    vector <Mat> baseArm = createBox(200, 50, 50);
    vector <Mat> midArm = createBox(200, 50, 50);
    vector <Mat> prisArm = createBox(200, 50, 50);

    Mat T0 = createHT(im.size().width / 2, im.size().height / 2, 0, CV_PI + 0.2, 0, 0);
    Mat T2, T3, T1;


    //Convert Degree
    J1 = J1 * (CV_PI / 180);
    J2 = J2 * (CV_PI / 180);
    J3 = J3 * (CV_PI / 180);

    //calculate transformation matrix
    T1 = T0 * createHT(0, 0, 0, 0, J1, 0);
    T2 = T1 * createHT(200, 0, 0, 0, J2, 0);
    T3 = T2 * createHT(200, -J4, 0, J3, 0, -CV_PI / 2);
    Mat dumT3 = T2 * createHT(200, 0, 0, 0, 0, 0);

    //transform each box
    transformBox(baseArm, T1);
    transformBox(midArm, T2);
    transformBox(prisArm, T3);

    //trnsform orgins
    drawUnitOr(im, T1);
    drawUnitOr(im, T2);
    drawUnitOr(im, T3);
    drawUnitOr(im, dumT3);

    //make the box
    makeBox(im, baseArm, CV_RGB(255, 0, 0));
    makeBox(im, midArm, CV_RGB(0, 255, 0));
    makeBox(im, prisArm, CV_RGB(0, 0, 255));

    //output image on screen
    cv::imshow("7825 Project", im);
   // cv::waitKey(0);
}

void Robot::animate(Mat& im) {
    Size image_size = Size(1000, 600);
    float baseAng = 0;
    float midAng = 0;
    float prizAng = 0;
    float prizPos = 0;
    int counter = 0;
    do {
        /////////////////////////////////////////////
        ///////Conditional statements to test///////
        ///////all joints//////////////////////////
        if (baseAng <= 360 * (CV_PI / 180)) {
            baseAng += 0.1;
        }
        else if (baseAng >= 360 * (CV_PI / 180) && midAng <= 360 * (CV_PI / 180)) {
            midAng += 0.1;
        }
        else if (baseAng >= 360 * (CV_PI / 180) && midAng >= 360 * (CV_PI / 180) && prizAng <= 360 * (CV_PI / 180)) {
            prizAng += 0.1;
        }
        else if (baseAng >= 360 * (CV_PI / 180) && midAng >= 360 * (CV_PI / 180) && prizAng >= 360 * (CV_PI / 180)) {
            if (counter <= 12)
                prizPos += 10;
            else if (counter > 12 && counter <= 25)
                prizPos -= 10;
            else {
                prizPos = prizPos;
            }
            counter++;
        }
        else {
            baseAng = baseAng;
            midAng = midAng;
            prizAng = prizAng;
            prizPos = prizPos;
        }
        //create new boxes
        vector <Mat> baseArm = createBox(200, 50, 50);
        vector <Mat> midArm = createBox(200, 50, 50);
        vector <Mat> prisArm = createBox(200, 50, 50);

        //calculate transformation matrix
        Mat T1 = createHT(im.size().width / 2, im.size().height / 2, 0, CV_PI + 0.2, 0, 0) * createHT(0, 0, 0, 0, baseAng, 0);;
        Mat T2 = T1 * createHT(200, 0, 0, 0, midAng, 0);
        Mat dumT3 = T2 * createHT(200, 0, 0, 0, 0, 0);
        Mat T3 = T2 * createHT(200, -prizPos, 0, prizAng, 0, -CV_PI / 2);

        //transform each box
        transformBox(baseArm, T1);
        transformBox(midArm, T2);
        transformBox(prisArm, T3);

        //trnsform orgins
        drawUnitOr(im, T1);
        drawUnitOr(im, T2);
        drawUnitOr(im, T3);
        drawUnitOr(im, dumT3);

        //make the box
        makeBox(im, baseArm, CV_RGB(255, 0, 0));
        makeBox(im, midArm, CV_RGB(0, 255, 0));
        makeBox(im, prisArm, CV_RGB(0, 0, 255));

        //output image on screen
        cv::imshow("7825 Project", im);

        //reset box matrix
        baseArm.clear();
        midArm.clear();
        prisArm.clear();

        //Clear image
        im = cv::Mat::zeros(image_size, CV_8UC3) + CV_RGB(60, 60, 60);

    } while (cv::waitKey(33) != 'q');

}

void Robot::ctraj(Mat& im, Mat initPos, Mat finPos, Mat initVel, Mat finVel, Mat timeSteps) {

    float tscal = 1;
    Size s = timeSteps.size();
    float t5, t4, t3, t2, t1, t0, a, b, c, d, e, f;

    Mat tt = Mat::zeros(cv::Size(6, s.height), CV_32FC1);
    Mat cons = Mat::zeros(cv::Size(6, 2), CV_32FC1);


    Mat A = 6 * (finPos - initPos) - 3 * (finVel + initVel) * tscal;
    Mat B = -15 * (finPos - initPos) + (8 * initVel + 7 * finVel) * tscal;
    Mat C = 10 * (finPos - initPos) - (6 * initVel + 4 * finVel) * tscal;
    Mat E = initVel * tscal;//as the t vector has been normalized
    Mat F = initPos;
    Mat D = Mat::zeros(size(A), CV_32FC1);


    cv::normalize(timeSteps, timeSteps, 1, 0, NORM_MINMAX);

    for (int row = 0; row < s.height; row++) {
        t5 = pow(timeSteps.at<float>(row, 0), 5);
        t4 = pow(timeSteps.at<float>(row, 0), 4);
        t3 = pow(timeSteps.at<float>(row, 0), 3);
        t2 = pow(timeSteps.at<float>(row, 0), 2);
        t1 = timeSteps.at<float>(row, 0);
        t0 = 1;

        tt.at<float>(row, 0) = t5;
        tt.at<float>(row, 1) = t4;
        tt.at<float>(row, 2) = t3;
        tt.at<float>(row, 3) = t2;
        tt.at<float>(row, 4) = t1;
        tt.at<float>(row, 5) = t0;
    }

    for (int row = 0; row < 2; row++) {
        a = A.at<float>(row, 0);
        b = B.at<float>(row, 0);
        c = C.at<float>(row, 0);
        d = D.at<float>(row, 0);
        e = E.at<float>(row, 0);
        f = F.at<float>(row, 0);

        cons.at<float>(row, 0) = a;
        cons.at<float>(row, 1) = b;
        cons.at<float>(row, 2) = c;
        cons.at<float>(row, 3) = d;
        cons.at<float>(row, 4) = e;
        cons.at<float>(row, 5) = f;
    }

    Mat coords = Mat::zeros(cv::Size(2, s.height), CV_32FC1);
    gemm(tt, cons.t(), 1.0, cv::Mat(), 0.0, coords);
    cout << coords;
    vector<float> joint_ang;

    for (int i = 0; i < s.height; i++) {
        joint_ang = ikine(coords.at<float>(i, 0), coords.at<float>(i, 1), 0, 0);
        manualDraw(im, joint_ang.at(0) * (180 / CV_PI), joint_ang.at(1) * (180 / CV_PI), 0, 0);
        cv::waitKey(10);
    }
}

Mat Robot::jtraj(Mat& im, Mat initPos, Mat finPos, Mat initVel, Mat finVel, Mat timeSteps) {

    float tscal = 1;
    Size s = timeSteps.size();
    float t5, t4, t3, t2, t1, t0, a, b, c, d, e, f;

    Mat tt = Mat::zeros(cv::Size(6, s.height), CV_32FC1);
    Mat cons = Mat::zeros(cv::Size(6, 2), CV_32FC1);


    Mat A = 6 * (finPos - initPos) - 3 * (finVel + initVel) * tscal;
    Mat B = -15 * (finPos - initPos) + (8 * initVel + 7 * finVel) * tscal;
    Mat C = 10 * (finPos - initPos) - (6 * initVel + 4 * finVel) * tscal;
    Mat E = initVel * tscal;//as the t vector has been normalized
    Mat F = initPos;
    Mat D = Mat::zeros(size(A), CV_32FC1);


    cv::normalize(timeSteps, timeSteps, 1, 0, NORM_MINMAX);

    for (int row = 0; row < s.height; row++) {
        t5 = pow(timeSteps.at<float>(row, 0), 5);
        t4 = pow(timeSteps.at<float>(row, 0), 4);
        t3 = pow(timeSteps.at<float>(row, 0), 3);
        t2 = pow(timeSteps.at<float>(row, 0), 2);
        t1 = timeSteps.at<float>(row, 0);
        t0 = 1;

        tt.at<float>(row, 0) = t5;
        tt.at<float>(row, 1) = t4;
        tt.at<float>(row, 2) = t3;
        tt.at<float>(row, 3) = t2;
        tt.at<float>(row, 4) = t1;
        tt.at<float>(row, 5) = t0;
    }
    for (int row = 0; row < 2; row++) {
        a = A.at<float>(row, 0);
        b = B.at<float>(row, 0);
        c = C.at<float>(row, 0);
        d = D.at<float>(row, 0);
        e = E.at<float>(row, 0);
        f = F.at<float>(row, 0);

        cons.at<float>(row, 0) = a;
        cons.at<float>(row, 1) = b;
        cons.at<float>(row, 2) = c;
        cons.at<float>(row, 3) = d;
        cons.at<float>(row, 4) = e;
        cons.at<float>(row, 5) = f;
    }

    Mat coords = Mat::zeros(cv::Size(2, s.height), CV_32FC1);
    gemm(tt, cons.t(), 1.0, cv::Mat(), 0.0, coords);
    return coords;
}

bool Robot :: readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

bool Robot:: readDetectorParameters(string filename, Ptr<aruco::DetectorParameters>& params) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

int Robot::detect(Mat& im, int argc, char* argv[]) {

   // Size image_size = Size(1000, 600);
    //image matrix to draw everything on
    //cv::Mat im = cv::Mat::zeros(image_size, CV_8UC3) + CV_RGB(60, 60, 60);
    //Robot robot;
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (argc < 6) {
        parser.printMessage();
        return 0;
    }

    int squaresX = parser.get<int>("w");
    int squaresY = parser.get<int>("h");
    float squareLength = parser.get<float>("sl");
    float markerLength = parser.get<float>("ml");
    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool refindStrategy = parser.has("rs");
    int camId = 0;// parser.get<int>("ci");

    String video;
    if (parser.has("v")) {
        video = parser.get<String>("v");
    }

    Mat camMatrix, distCoeffs;
    if (parser.has("c")) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if (!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if (parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if (!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    VideoCapture inputVideo;
    int waitTime;
    if (!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    }
    else {
        inputVideo.open(camId);
        waitTime = 10;
    }

    float axisLength = 0.5f * ((float)min(squaresX, squaresY) * (squareLength));

    // create charuco board object
    Ptr<aruco::CharucoBoard> charucoboard =
        aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
    Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();

    double totalTime = 0;
    int totalIterations = 0;

    while (inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector< int > markerIds, charucoIds;
        vector< vector< Point2f > > markerCorners, rejectedMarkers;
        vector< Point2f > charucoCorners;
        Vec3d rvec, tvec;

        // detect markers
        aruco::detectMarkers(image, dictionary, markerCorners, markerIds, detectorParams,
            rejectedMarkers);

        // refind strategy to detect more markers
        if (refindStrategy)
            aruco::refineDetectedMarkers(image, board, markerCorners, markerIds, rejectedMarkers,
                camMatrix, distCoeffs);

        // interpolate charuco corners
        int interpolatedCorners = 0;
        if (markerIds.size() > 0)
            interpolatedCorners =
            aruco::interpolateCornersCharuco(markerCorners, markerIds, image, charucoboard,
                charucoCorners, charucoIds, camMatrix, distCoeffs);

        // estimate charuco board pose
        bool validPose = false;
        if (camMatrix.total() != 0)

            validPose = aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, charucoboard,
                camMatrix, distCoeffs, rvec, tvec);
        cout << tvec << "\n";

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if (totalIterations % 30 == 0) {
            //cout << "Detection Time = " << currentTime * 1000 << " ms "
              //  << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

        // draw results
        image.copyTo(imageCopy);
        if (markerIds.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, markerCorners);
        }

        if (showRejected && rejectedMarkers.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejectedMarkers, noArray(), Scalar(100, 0, 255));

        if (interpolatedCorners > 0) {
            Scalar color;
            color = Scalar(255, 0, 0);
            aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, color);
        }

        if (validPose)
            aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);

        imshow("out", imageCopy);

        vector<float> q = ikine(-1000 * tvec[0], -1000 * tvec[1], 0, 0);
        manualDraw(im, q.at(0) * (180 / CV_PI), q.at(1) * (180 / CV_PI), 0, -200 * tvec[2]);

        char key = (char)waitKey(waitTime);
        if (key == 27) break;
    }

    return 0;
}

bool Robot::saveCameraParams(const string& filename, Size imageSize, float aspectRatio, int flags, const Mat& cameraMatrix, const Mat& distCoeffs, double totalAvgErr) {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
        return false;

    time_t tt;
    time(&tt);
    struct tm* t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    if (flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

    if (flags != 0) {
        sprintf(buf, "flags: %s%s%s%s",
            flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;

    return true;
}

int Robot::Calibrate(int argc, char* argv[]) {
    CommandLineParser parser(argc, argv, keys1);
    parser.about(about);

     if (argc < 7) {
        parser.printMessage();
        return 0;
     }

    int squaresX = 5;// parser.get<int>("w");
    int squaresY = 7;// parser.get<int>("h");
    float squareLength = 0.04; //parser.get<float>("sl");
    float markerLength = 0.02;// parser.get<float>("ml");
    int dictionaryId = 10;//parser.get<int>("d");
    string outputFile = "\calib.txt";// parser.get<string>(0);

    bool showChessboardCorners = false;// parser.get<bool>("sc");

    int calibrationFlags = 0;
    float aspectRatio = 1;
    if (parser.has("a")) {
        calibrationFlags |= CALIB_FIX_ASPECT_RATIO;
        aspectRatio = parser.get<float>("a");
    }
    if (parser.get<bool>("zt")) calibrationFlags |= CALIB_ZERO_TANGENT_DIST;
    if (parser.get<bool>("pc")) calibrationFlags |= CALIB_FIX_PRINCIPAL_POINT;

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if (parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if (!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }

    bool refindStrategy = parser.get<bool>("rs");
    int camId = parser.get<int>("ci");
    String video;

    if (parser.has("v")) {
        video = parser.get<String>("v");
    }

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    VideoCapture inputVideo;
    int waitTime;
    if (!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    }
    else {
        inputVideo.open(camId);
        waitTime = 10;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    // create charuco board object
    Ptr<aruco::CharucoBoard> charucoboard =
        aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
    Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();

    // collect data from each frame
    vector< vector< vector< Point2f > > > allCorners;
    vector< vector< int > > allIds;
    vector< Mat > allImgs;
    Size imgSize;

    while (inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;

        // detect markers
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

        // refind strategy to detect more markers
        if (refindStrategy) aruco::refineDetectedMarkers(image, board, corners, ids, rejected);

        // interpolate charuco corners
        Mat currentCharucoCorners, currentCharucoIds;
        if (ids.size() > 0)
            aruco::interpolateCornersCharuco(corners, ids, image, charucoboard, currentCharucoCorners,
                currentCharucoIds);

        // draw results
        image.copyTo(imageCopy);
        if (ids.size() > 0) aruco::drawDetectedMarkers(imageCopy, corners);

        if (currentCharucoCorners.total() > 0)
            aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, currentCharucoIds);

        putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
            Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if (key == 27) break;
        if (key == 'c' && ids.size() > 0) {
            cout << "Frame captured" << endl;
            allCorners.push_back(corners);
            allIds.push_back(ids);
            allImgs.push_back(image);
            imgSize = image.size();
        }
    }

    if (allIds.size() < 1) {
        cerr << "Not enough captures for calibration" << endl;
        return 0;
    }

    Mat cameraMatrix, distCoeffs;
    vector< Mat > rvecs, tvecs;
    double repError;

    if (calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at< double >(0, 0) = aspectRatio;
    }

    // prepare data for calibration
    vector< vector< Point2f > > allCornersConcatenated;
    vector< int > allIdsConcatenated;
    vector< int > markerCounterPerFrame;
    markerCounterPerFrame.reserve(allCorners.size());
    for (unsigned int i = 0; i < allCorners.size(); i++) {
        markerCounterPerFrame.push_back((int)allCorners[i].size());
        for (unsigned int j = 0; j < allCorners[i].size(); j++) {
            allCornersConcatenated.push_back(allCorners[i][j]);
            allIdsConcatenated.push_back(allIds[i][j]);
        }
    }

    // calibrate camera using aruco markers
    double arucoRepErr;
    arucoRepErr = aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
        markerCounterPerFrame, board, imgSize, cameraMatrix,
        distCoeffs, noArray(), noArray(), calibrationFlags);

    // prepare data for charuco calibration
    int nFrames = (int)allCorners.size();
    vector< Mat > allCharucoCorners;
    vector< Mat > allCharucoIds;
    vector< Mat > filteredImages;
    allCharucoCorners.reserve(nFrames);
    allCharucoIds.reserve(nFrames);

    for (int i = 0; i < nFrames; i++) {
        // interpolate using camera parameters
        Mat currentCharucoCorners, currentCharucoIds;
        aruco::interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], charucoboard,
            currentCharucoCorners, currentCharucoIds, cameraMatrix,
            distCoeffs);

        allCharucoCorners.push_back(currentCharucoCorners);
        allCharucoIds.push_back(currentCharucoIds);
        filteredImages.push_back(allImgs[i]);
    }

    if (allCharucoCorners.size() < 4) {
        cerr << "Not enough corners for calibration" << endl;
        return 0;
    }

    // calibrate camera using charuco
    repError =
        aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoboard, imgSize,
            cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);

    bool saveOk = saveCameraParams(outputFile, imgSize, aspectRatio, calibrationFlags,
        cameraMatrix, distCoeffs, repError);
    if (!saveOk) {
        cerr << "Cannot save output file" << endl;
        return 0;
    }

    cout << "Rep Error: " << repError << endl;
    cout << "Rep Error Aruco: " << arucoRepErr << endl;
    cout << "Calibration saved to " << outputFile << endl;

    // show interpolated charuco corners for debugging
    if (showChessboardCorners) {
        for (unsigned int frame = 0; frame < filteredImages.size(); frame++) {
            Mat imageCopy = filteredImages[frame].clone();
            if (allIds[frame].size() > 0) {

                if (allCharucoCorners[frame].total() > 0) {
                    aruco::drawDetectedCornersCharuco(imageCopy, allCharucoCorners[frame],
                        allCharucoIds[frame]);
                }
            }

            imshow("out", imageCopy);
            char key = (char)waitKey(0);
            if (key == 27) break;
        }
    }

    return 0;
}

void Robot::YOLO(Mat& showRobot) {
    //YOLO Initialization
    Size network_input_image_size = Size(416, 416);  	
    vector<string> output_layer_name;

    // Load vector of class names (objects trained to identify)  	
    vector<string> classnames;  	
    string load_str; 
    ifstream filename("coco.names");  	
    while (std::getline(filename, load_str)) {
        classnames.push_back(load_str);
    }

    // Load YoloV3 DNN configuration and weights 
    cv::dnn::Net net = readNetFromDarknet("yolov3.cfg", "yolov3.weights");  	
    net.setPreferableBackend(DNN_BACKEND_OPENCV);  	
    net.setPreferableTarget(DNN_TARGET_CPU);
    

    // Get layer names and output layer indexes  	
    vector<string> layer_name = net.getLayerNames(); 
 
    
    vector<int> output_layer_index = net.getUnconnectedOutLayers();
    cout << output_layer_index.size();
    for (int out_index = 0; out_index < output_layer_index.size(); out_index++) {
        output_layer_name.push_back(layer_name.at(output_layer_index.at(out_index) - 1));
    }

  
    //YOLO Operation
    Mat input_blob;
    Mat im;
    vector<Mat> network_output;

    //webcam
    VideoCapture inputVideo;
    inputVideo.open(0);
    waitKey(10);
    while (inputVideo.grab()) {
        inputVideo.retrieve(im);

        // Setup 4D Tensor - scale and resize input image to fit network trained data  
        blobFromImage(im, input_blob, 1 / 255.0, network_input_image_size, Scalar(0, 0, 0), true, false);
        // Input the image to network  	 	
        net.setInput(input_blob);

        // Runs the forward pass  
        net.forward(network_output, output_layer_name);

        // YOLO Process Results

        // Pull out all the found objects with confidence and bounding boxes  	
        vector<int> id;
        vector<float> conf;
        vector<Rect> bbox;

        float confidence_threshold = 0.6;
        float nms_threshold = 0.35;

        // Loop over each output layer (x3 YOLOv3) 
        for (int output_index = 0; output_index < network_output.size(); output_index++) {
            int detections_in_layer = network_output[output_index].rows;
            int output_data_offset = network_output[output_index].cols;
            // Loop over each detection  
            for (int detect_index = 0; detect_index < detections_in_layer; detect_index++) {
                Point max_class;
                double max_conf;

                // network_output = bbox (center (x, y), width, height), bbox confidence, 80x class confidence 
                // index to a single result 
                float* data = (float*)network_output[output_index].data + detect_index * output_data_offset;
                // Get all class scores (confidence per class) (x80 YOLOv3) 
                Mat class_conf = network_output[output_index].row(detect_index).colRange(5, network_output[output_index].cols);

                // Find class with highest confidence  	 	 	
                minMaxLoc(class_conf, 0, &max_conf, 0, &max_class);

                // If high enough confidence then save  	 	 	
                if (max_conf > confidence_threshold)
                {
                    Point bbox_center;
                    Size bbox_size;

                    // Rescale bbox to image size in pixels (from 0..1)  	 	 	 	
                    bbox_center.x = (int)(data[0] * im.size().width);
                    bbox_center.y = (int)(data[1] * im.size().height);
                    bbox_size.width = (int)(data[2] * im.size().width);
                    bbox_size.height = (int)(data[3] * im.size().height);

                    // save 
                    id.push_back(max_class.x);
                    conf.push_back((float)max_conf);
                    bbox.push_back(Rect(bbox_center.x - bbox_size.width / 2, bbox_center.y - bbox_size.height / 2, bbox_size.width, bbox_size.height));
                }
            }
        }
        // Non Maximum Suppression - remove overlapping boxes   	
        vector<int> found_index;
        dnn::NMSBoxes(bbox, conf, confidence_threshold, nms_threshold, found_index);
        vector<float> q;
        //YOLO Results Display
        for (int object_index = 0; object_index < found_index.size(); object_index++)
        {
            int index = found_index.at(object_index);
            rectangle(im, bbox.at(index), Scalar(255, 255, 0));

            string str = format("%.2f: ", conf.at(index));
            str = str + classnames.at(id.at(index));

            if (classnames.at(id.at(index)) == "banana") {
                //scara stuff
                float z = 0.002 * bbox.at(index).height * bbox.at(index).width;
                cout << classnames.at(id.at(index))<<"\n";
               cout << "x = "<< bbox.at(index).x <<", y = " << bbox.at(index).y << ", z = " << z << "\n";
                q = ikine(0.75*bbox.at(index).x, 0.75*bbox.at(index).y, 0, 0);
                manualDraw(showRobot, q.at(0) * (180 / CV_PI), q.at(1) * (180 / CV_PI),0, z);
                waitKey(1);
            }         
            putText(im, str, Point(bbox.at(index).x, bbox.at(index).y - 10), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255, 255, 255), 2);
        }

        // Time per layer and overall time  	
        vector<double> layer_proc_time;
        double proc_time = net.getPerfProfile(layer_proc_time) / getTickFrequency();
        putText(im, format("Proc Time: %.2f ", proc_time), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255, 255, 255), 2);

        imshow("DNN", im);
        waitKey(5);
    }
}

Robot::~Robot()
{

}