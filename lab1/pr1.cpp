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
    cv::Mat img = cv::imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/picture.jpg");
    if (!img.data) {
        std::cout << "Could not open or find the image" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string text = format("Width: %d, Height: %d", img.cols, img.rows);

    putText(img, text, Point(25, 25), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);

    cv::line(img, Point(80, 420), Point(200, 420), Scalar(0, 255, 0), 2);

    cv::circle(img, Point(250, 250), 30, Scalar(0, 0, 255), -1, 8, 0);

    cv::circle(img, Point(400,50), 30, Scalar(0, 255, 255), 2);

    cv::rectangle(img, Point(50, 50), Point(100, 100), Scalar(255, 0, 0), 3);

    cv::rectangle(img, Point(350, 350), Point(500, 500), Scalar(0, 136, 255), -1, 8, 0);

    cv::imshow("First OpenCV Application", img);

    cv::Mat seg1 = img(Rect(0, 0, img.rows/2, img.rows/2));
    cv::imshow("1 segment", seg1);

    cv::Mat seg2 = img(Rect(0, img.cols / 2, img.rows / 2, img.rows / 2));
    cv::imshow("2 segment", seg2);

    cv::Mat seg3 = img(Rect(img.cols/2, 0, img.rows / 2, img.rows / 2));
    cv::imshow("3 segment", seg3);

    cv::Mat seg4 = img(Rect(img.cols / 2, img.cols / 2, img.rows / 2, img.rows / 2));
    cv::imshow("4 segment", seg4);

    cv::Mat mask(img.size(), CV_8UC1, Scalar(0));
    cv::rectangle(mask, Point(170, 170), Point(340, 340), Scalar(255, 255, 255), -1);

    Mat mmask;
    img.copyTo(mmask, mask);
    cv::imshow("Mask", mmask);



    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
