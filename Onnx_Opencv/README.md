# RUN
```bash
mkdir build
cd build
cmake ..
make
```
GPU加速关键代码
```c++
auto net = dnn::readNetFromONNX("resnet50.onnx");
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```
