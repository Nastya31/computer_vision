#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using namespace std;
using namespace cv;

int main()
{
    // Задание 1
    Mat imgg = cv::imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/figure.jpg", IMREAD_COLOR);
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
    vector<vector<Point>> contour;
    findContours(edges, contour, RETR_TREE, CHAIN_APPROX_TC89_L1);
    imshow("Lines", edges);

    Mat draw = Mat::zeros(edges.size(), CV_8UC3);
    size_t i = 0;
    for (size_t i = 0; i < contour.size(); i++) {
        vector<Point> approx;
        approxPolyDP(contour[i], approx, arcLength(contour[i], true) * 0.01, true);
        size_t number = approx.size();

        string text;
        Scalar color;
        Moments center = moments(contour[i]);
        if (number == 4)
        {
            text = "Square";
            color = Scalar(255, 0, 0);
        }
        if (number == 3) 
        {
            text = "Triangle";
            color = Scalar(0, 255, 0);
        }
        if (number != 4 && number != 3)
        {
            text = "Circle";
            color = Scalar(0, 0, 255);
        }
        drawContours(draw, contour, i, color, 2);
        Point text_coord(center.m10 / center.m00 - 40, center.m01 / center.m00);
        putText(draw, text, text_coord, FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    }
    imshow("Contours", draw);

    //Задание 2
    Mat imgg2 = cv::imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/book.jpg", IMREAD_COLOR);
    if (!imgg2.data) {
        std::cout << "Could not open or find the image" << std::endl;
        exit(EXIT_FAILURE);
    }
    Mat gray2;
    cvtColor(imgg2, gray2, COLOR_BGR2GRAY);

    Mat image_blurred2;
    GaussianBlur(gray2, image_blurred2, Size(3, 3), 0);
    Mat edges2;
    Canny(gray2, edges2, 50, 200);
    resize(edges2, edges2, Size(900, 700), INTER_LINEAR);
    imshow("Text", edges2);

    waitKey(0);
    destroyAllWindows();
    return 0;
}