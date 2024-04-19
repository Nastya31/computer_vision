#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;


void detectCard(string& cardName, Mat& card, vector<Mat>& cardsImages, vector<string>& cardsNames, vector<Mat>& cardsDescriptors, vector<vector<KeyPoint>>& cardsKeypoints) {

	Mat cardDescriptors;
	vector<KeyPoint> cardKeypoints;
	Ptr<ORB> detector = ORB::create();
	Ptr<BFMatcher> matcher = BFMatcher::create();
	detector->detectAndCompute(card, noArray(), cardKeypoints, cardDescriptors);

	if (cardDescriptors.empty()) {
		cardName = "";
		return;
	}

	int maxI = -1;
	int maxCount = 0;

	for (int i = 0; i < cardsImages.size(); i++) {

		if (cardsDescriptors[i].empty()) {
			continue;
		}

		vector<vector<DMatch>> knn_matches;

		matcher->knnMatch(cardsDescriptors[i], cardDescriptors, knn_matches, 3);

		vector<DMatch> correct;

		for (size_t i = 0; i < knn_matches.size(); i++) {
			if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
				correct.push_back(knn_matches[i][0]);
			}
		}

		if (maxCount < correct.size()) {
			maxCount = static_cast<int>(correct.size());
			maxI = i;
		}
	}

	if (maxI == -1) {
		cardName = "";
	}
	else {
		cardName = cardsNames[maxI];
	}
}


int main() {

	vector<Mat> card_img;
	vector<string> card_nm;
	vector<Mat> card_descr;
	vector<vector<KeyPoint>> card_key;

	Mat card;
	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/TusB.png");
	card_img.push_back(card);
	card_nm.push_back("Tus_buby");

	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/10ch.png");
	card_img.push_back(card);
	card_nm.push_back("10_heart");

	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/DamaK.png");
	card_img.push_back(card);
	card_nm.push_back("Dama_kresty");

	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/DamaV.png");
	card_img.push_back(card);
	card_nm.push_back("Dama_viny");

	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/KingK.png");
	card_img.push_back(card);
	card_nm.push_back("King_kresty");

	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/KingV.png");
	card_img.push_back(card);
	card_nm.push_back("King_viny");

	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/ValetV.png");
	card_img.push_back(card);
	card_nm.push_back("Valet_viny");


	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/TusV.png");
	card_img.push_back(card);
	card_nm.push_back("Tus_viny");

	card = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/cards/DamaCh.png");
	card_img.push_back(card);
	card_nm.push_back("Dama_heart");

	Ptr<ORB> detector = ORB::create();

	for (int i = 0; i < card_img.size(); i++) {
		Mat dis;
		vector<KeyPoint> keys;
		detector->detectAndCompute(card_img[i], noArray(), keys, dis);
		card_key.push_back(keys);
		card_descr.push_back(dis);
	}

	Mat image = imread("C:/Users/solov/source/repos/OpenCVProjects/OpenCVProjects/photo_cards2.jpg");

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Mat blur, thresh, canny;
	GaussianBlur(image, blur, Size(15, 15), 3);
	Canny(blur, canny, 120, 120);
	findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours) {
		vector<Point> contoursPoly;

		approxPolyDP(contour, contoursPoly, 1, true);

		RotatedRect cardRect = minAreaRect(contoursPoly);

		if (cardRect.size.width < 100 || cardRect.size.height < 100) {
			continue;
		}

		Mat card;
		string cardName;

		Mat rotatedMatrix, rotatedImage;
		rotatedMatrix = getRotationMatrix2D(cardRect.center, cardRect.angle, 1.0);
		warpAffine(image, rotatedImage, rotatedMatrix, image.size(), INTER_CUBIC);
		getRectSubPix(rotatedImage, cardRect.size, cardRect.center, card);

		rotate(card, card, ROTATE_180);

		if (card.size[0] < card.size[1]) {
			rotate(card, card, ROTATE_90_CLOCKWISE);
		}

		detectCard(cardName, card, card_img, card_nm, card_descr, card_key);

		if (cardName != "") {
			Point2f boxPoints[4];
			cardRect.points(boxPoints);

			for (int j = 0; j < 4; j++) {
				line(image, boxPoints[j], boxPoints[(j + 1) % 4], Scalar(255, 0, 0), 4, LINE_AA);
			}
			putText(image, cardName, cardRect.center, 1, 2, Scalar(255, 0, 0), 2);
		}
	}
	imshow("Cards", image);
	waitKey(0);
}