# RUN
- 首先运行resnet_onnx.py生成resnet50.onnx文件
- 然后，编译c++文件
```bash
mkdir build
cd build
cmake ..
make
```
- 最后，将资源文件拷贝到build文件夹，运行即可
# GPU加速关键代码
```c++
auto net = dnn::readNetFromONNX("resnet50.onnx");
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```
![img](https://github.com/jjw-DL/Model_Deployment/blob/master/Onnx_Opencv/result.jpg)
