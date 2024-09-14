cmake_minimum_required(VERSION 3.29)
project(Demo1_6)

set(CMAKE_CXX_STANDARD 20)

add_executable(Demo1_6 main.cpp
        rasterizer.h
        Triangle.h
        rasterizer.cpp
        Triangle.cpp)

set(OpenCV_DIR $ENV{OPENCV_DIR})
include_directories($ENV{EIGEN_DIR})

find_package(OpenCV REQUIRED)
target_link_libraries(Demo1_6 ${OpenCV_LIBS})

