cmake_minimum_required(VERSION 2.8.12)
project(kNN-classifier)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")


IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug mode")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O0")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
ENDIF()

IF(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release mode")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ENDIF()

include_directories(extern/eigen)

add_subdirectory(extern/pybind11)

pybind11_add_module(kNN
                    src/kNN-pybind11.cpp
                    src/kNN.cpp
                    src/pca.cpp
                    src/eigen.cpp)

set(CMAKE_INSTALL_PREFIX
        ${PROJECT_SOURCE_DIR}
        CACHE PATH "Python custom modules directory" FORCE)

message(CMAKE_INSTALL_PREFIX = "${CMAKE_INSTALL_PREFIX}")

install(TARGETS kNN DESTINATION ${CMAKE_INSTALL_PREFIX}/notebooks)

add_executable(kNN-classifier
               src/main.cpp
               src/kNN.cpp
               src/eigen.cpp
               src/pca.cpp)