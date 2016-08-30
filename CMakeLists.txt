project(SemiCRFSegmentation)
cmake_minimum_required(VERSION 2.7 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3")
endif()

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${N3L_INCLUDE_DIR})

add_subdirectory(src)
