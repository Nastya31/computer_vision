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


int main()
{
        Mat imgg = cv::imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/face.png", IMREAD_COLOR);
        if (!imgg.data) {
            std::cout << "Could not open or find the image" << std::endl;
            exit(EXIT_FAILURE);
        }
        imshow("Not gray", imgg);

        Mat gray;
        cvtColor(imgg, gray, COLOR_BGR2GRAY);

        cv::CascadeClassifier face_cascade_eye;
        if (!face_cascade_eye.load(samples::findFile("haarcascade_eye.xml"))) {
            cout << "Error" << endl;
        }
        vector<Rect> eyes;
       face_cascade_eye.detectMultiScale(gray, eyes, 1.1, 5);

       for (const auto& eye : eyes)
       {
           rectangle(imgg, eye, Scalar(255, 0, 0),1);
       }

       cv::CascadeClassifier face_cascade_smile;
       if (!face_cascade_smile.load(samples::findFile("haarcascade_smile.xml"))) {
           cout << "Error" << endl;
       }

       vector<Rect> smiles;
       face_cascade_smile.detectMultiScale(gray, smiles, 1.1, 35);

       for (const auto& smile : smiles)
       {
           rectangle(imgg, smile, Scalar(0, 255, 0), 1);
       }

       cv::CascadeClassifier face_cascade_face;
       if (!face_cascade_face.load(samples::findFile("haarcascade_frontalface_default.xml"))) {
           cout << "Error" << endl;
       }
       vector<Rect> faces;
       face_cascade_face.detectMultiScale(gray, faces, 1.1, 5);

       for (const auto& face : faces)
       {
           rectangle(imgg, face, Scalar(0, 0, 255), 1);
       }
       imshow("Face", imgg);

    waitKey(0);
    destroyAllWindows();
    return 0;
}