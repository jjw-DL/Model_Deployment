# RUN
- 首先下载模型权重：https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
- 将yolov4.weights、classes.txt、demo.mp4、yolov4.cfg文件放入build文件夹（防止找不到资源）
```bash
mkdir build
cd build
cmake ..
make
```
GPU加速关键代码
```c++
auto net = cv::dnn::readNetFromDarknet("yolov4.cfg", "yolov4.weights");
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```

 
