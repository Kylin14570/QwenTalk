cmake_minimum_required(VERSION 3.0.0)

project(QwenTalk)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(QwenTalk main.cpp ${SRC_FILES})