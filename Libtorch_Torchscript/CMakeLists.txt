cmake_minimum_required(VERSION 3.0)
project(predict_demo)

set(Torch_DIR /home/ggj/pytorch/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(classifier prediction.cpp)
target_link_libraries(classifier ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# set C++14 to compile PyTorch
set_property(TARGET classifier PROPERTY CXX_STANDARD 14)
