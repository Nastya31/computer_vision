#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using namespace cv;
using namespace std;


int main() {
    VideoCapture cap("video.mp4");
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("white_square.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));
   

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while (1) {

        Mat imgg;
        cap >> imgg;

        if (imgg.empty())
            break;
        
        Mat frame_HSV;
        cvtColor(imgg, frame_HSV, COLOR_BGR2HSV);

        Mat white;
        inRange(frame_HSV, Scalar(95, 12, 138), Scalar(255, 50, 255), white);

        Mat edges;
        Canny(white, edges, 50, 200);
        
        Mat image_blurred;
        GaussianBlur(edges, image_blurred, Size(5, 5), 0);
        
        vector<vector<Point>> contour;
        vector<Vec4i> hierarchy;
        findContours(image_blurred, contour, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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
            drawContours(imgg, contour, i, color, 1, LINE_8, hierarchy, 0);
            Point text_coord(center.m10 / center.m00 - 40, center.m01 / center.m00);
            putText(imgg, text, text_coord, FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        }
        imshow("White square", imgg);
        video.write(imgg);
        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}