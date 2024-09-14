cmake_minimum_required(VERSION 3.29)
project(Demo1_6)

set(CMAKE_CXX_STANDARD 20)

add_executable(Demo1_6 main.cpp
        rasterizer.h
        Triangle.h
        rasterizer.cpp
        Triangle.cpp)

set(OpenCV_DIR E:\\CPPLib\\Opencv4.10)
include_directories("E:\\CPPLib\\eigen-3.4.0")
#set(OpenCV_LIBS E:/CPPLib/OpenCV_4.10/MingwBulid/install/x64/mingw/bin)

find_package(OpenCV REQUIRED)
target_link_libraries(Demo1_6 ${OpenCV_LIBS})

