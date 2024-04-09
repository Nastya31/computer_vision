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
    VideoCapture cap("ZUA.mp4");
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("face.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));
   

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while (1) {

        Mat imgg;
        cap >> imgg;

        if (imgg.empty())
            break;
        
        Mat gray;
        cvtColor(imgg, gray, COLOR_BGR2GRAY);
           
        cv::CascadeClassifier face_cascade_eye;
        if (!face_cascade_eye.load(samples::findFile("haarcascade_eye.xml"))) {
            cout << "Error" << endl;
        }
        vector<Rect> eyes;
        face_cascade_eye.detectMultiScale(gray, eyes, 1.3, 35);

        for (const auto& eye : eyes)
        {
            rectangle(imgg, eye, Scalar(255, 0, 0), 1);
        }

        cv::CascadeClassifier face_cascade_smile;
        if (!face_cascade_smile.load(samples::findFile("haarcascade_smile.xml"))) {
            cout << "Error" << endl;
        }

        vector<Rect> smiles;
        face_cascade_smile.detectMultiScale(gray, smiles, 1.9, 21);

        for (const auto& smile : smiles)
        {
            rectangle(imgg, smile, Scalar(0, 255, 0), 1);
        }

        cv::CascadeClassifier face_cascade_face;
        if (!face_cascade_face.load(samples::findFile("haarcascade_frontalface_default.xml"))) {
            cout << "Error" << endl;
        }
        vector<Rect> faces;
        face_cascade_face.detectMultiScale(gray, faces, 1.1, 15);

        for (const auto& face : faces)
        {
            rectangle(imgg, face, Scalar(0, 0, 255), 1);
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