#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

int softmax(const cv::Mat &src, cv::Mat &dst) {
  float max = 0.0;
  float sum = 0.0;
  max = *max_element(src.begin<float>(), src.end<float>());
  // cv::exp（src, dst）计算src中每一元素的指数并放入dst中
  cv::exp((src - max), dst);
  sum = cv::sum(dst)[0];
  dst /= sum;
  return 0;
}

int main(int argc, char ** argv) {

	//  加载labels
	std::vector<std::string> classes;
	std::string file = "classification_classes_ILSVRC2012.txt";
	std::ifstream ifs(file.c_str());
	if (!ifs.is_open())
		CV_Error(Error::StsError, "File " + file + " not found");
	std::string line;
	while (std::getline(ifs, line))
	{
		classes.push_back(line);
	}

	// 加载模型并载入CUDA
	auto net = dnn::readNetFromONNX("resnet50.onnx");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	// 创建窗口
	static const std::string kWinName = "Deep learning image classification in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);

	// 读取图片
	cv::Mat image;
	cv::Mat blob;
	image = cv::imread("squirrel_cls.jpg");
  cv::Mat image_show = image.clone();
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
	dnn::blobFromImage(image, blob, 1, Size(224, 224), Scalar(0.485, 0.456, 0.406), true, false);
	divide(blob, Scalar(0.229, 0.224, 0.225), blob);
	
	// 前向推理
	net.setInput(blob);
	Mat prob = net.forward();
	cout << "prob is " << prob.size() << endl;
	
  // 获取类别和置信度
	Point classIdPoint;
	double confidence;
  Mat prob_soft;
  softmax(prob, prob_soft);
  /*
    Mat Mat::reshape(int cn, int rows=0) const
    cn: 表示通道数(channels), 如果设为0，则表示保持通道数不变，否则则变为设置的通道数。
    rows: 表示矩阵行数。 如果设为0，则表示保持原有的行数不变，否则则变为设置的行数。
  */
	minMaxLoc(prob_soft.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
  // 显示
	std::vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = format("Inference time: %.2f ms", t);
	putText(image_show, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

	label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() : classes[classId].c_str()), confidence);
	putText(image_show, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

	imshow(kWinName, image_show);
  cv::imwrite("result.jpg",image_show);
  waitKey(0);

	return 0;

}

