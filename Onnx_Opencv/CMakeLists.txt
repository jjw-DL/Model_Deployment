cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(YOLOv4OpencvCUDA)
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV 4 REQUIRED)

include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(classification classification.cpp)
target_link_libraries(classification ${OpenCV_LIBS})
set_property(TARGET classification PROPERTY CXX_STANDARD 14)