# 深度学习模型部署基础（模型都较为简单常用，主要学习部署模式）
## Libtorch_Torchscript
- pytorch中利用Torchscript生成.pt模型，然后利用Libtorch的c++前端，调用模型进行图片预测

## Onnx_Opencv
- 在pytorch中导出resnet50的onnx模型，在c++环境中通过Opencv加载onnx模型，并利用CUDA进行加速

## Opencv_yolov4_Cuda
- 在c++环境中，通过Opencv加载yolov4进行目标检测，通过CUDA进行加速

## PytorchExtNcRelu
-  pytorch的C++拓展示例
- 参考博客：https://zhuanlan.zhihu.com/p/350651297

## PytorchExtNcReluCuda
-  C++/CUDA 混合的拓展
- 参考博客：https://zhuanlan.zhihu.com/p/350849116

