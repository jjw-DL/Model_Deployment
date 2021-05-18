cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(YOLOv4OpencvCUDA)

find_package(OpenCV 4 REQUIRED)

include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(yolov4_opencv_dnn_cuda yolov4_opencv_dnn_cuda.cpp)
target_link_libraries(yolov4_opencv_dnn_cuda ${OpenCV_LIBS})
set_property(TARGET yolov4_opencv_dnn_cuda PROPERTY CXX_STANDARD 14)