cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(ok VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(ok ok.cc)
target_link_libraries(ok Eigen3::Eigen)

