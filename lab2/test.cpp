#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using namespace cv;

int main()
{
    Mat imgg = cv::imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/island.jpeg", IMREAD_COLOR);
    if (!imgg.data) {
        std::cout << "Could not open or find the image" << std::endl;
        exit(EXIT_FAILURE);
    }
    imshow("Not gray", imgg);

    Mat gray;
    cvtColor(imgg, gray, COLOR_BGR2GRAY);

    Mat image_blurred;
    GaussianBlur(gray, image_blurred, Size(3, 3), 0);
    imshow("Gray", gray);

    Mat edges;
    Canny(gray, edges, 50, 200);

    std::vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 150, 10, 250);
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(imgg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 1, LINE_AA);
    }
    imshow("Lines", imgg);

    std::vector<Vec3f> circles;
    HoughCircles(edges, circles, HOUGH_GRADIENT, 1, 750, 200, 10, 10, 100);
    for (size_t i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(imgg, center, radius, Scalar(0, 255, 0), 2, 8, 0);
    }
    imshow("Lines", imgg);

    //Различные цвета
    Mat HSV;
    cvtColor(imgg, HSV, COLOR_BGR2HSV);
    imshow("HSV", HSV);

    Mat Lab;
    cvtColor(imgg, Lab, COLOR_BGR2Lab);
    imshow("Lab", Lab);

    Mat YUV;
    cvtColor(imgg, YUV, COLOR_BGR2YUV);
    imshow("YUV", YUV);

    Mat XYZ;
    cvtColor(imgg, XYZ, COLOR_BGR2XYZ);
    imshow("XYZ", XYZ);


    waitKey(0);
    destroyAllWindows();
    return 0;
}