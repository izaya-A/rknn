cmake_minimum_required(VERSION 3.0.0)

project(rknn_yolov8_demo)

set(CMAKE_CXX_STANDARD 11)

include_directories(./)
include_directories(/usr/local/include/opencv4)
link_directories(/usr/local/lib)
link_directories(/home/firefly/lff/new_pjt/build)

add_executable(rknn_yolov8_demo main.cpp postprocess.cc)
target_link_libraries(rknn_yolov8_demo opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc rknn_api rga rknnrt mpimmz )
target_link_libraries(rknn_yolov8_demo rknn_api rga rknnrt mpimmz)
