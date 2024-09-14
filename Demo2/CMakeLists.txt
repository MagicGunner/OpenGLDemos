cmake_minimum_required(VERSION 3.29)
project(Demo2)

set(CMAKE_CXX_STANDARD 20)

add_executable(Demo2 main.cpp
        Triangle.h
        Triangle.cpp
        rasterizer.h
        rasterizer.cpp
        global.h)

set(OpenCV_Dir $ENV{OPENCV_DIR})
include_directories($ENV{EIGEN_DIR})

find_package(OpenCV REQUIRED)
target_link_libraries(Demo2 ${OpenCV_LIBS})
