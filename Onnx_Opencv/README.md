# RUN
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
