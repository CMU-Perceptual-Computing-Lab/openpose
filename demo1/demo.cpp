#include <opencv2/opencv.hpp>
using namespace cv;

// 图像展示
int main() {
	Mat img = imread("1.jpg");
	imshow("Pic:", img);
	waitKey(6000);
	return 0;
}
