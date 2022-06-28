cmake_minimum_required(VERSION 3.5.1)
project(object_detection)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-march=native -Wall")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(MLPACK REQUIRED)



include_directories(
  ${MLPACK_INCLUDE_DIRS}
)

add_executable(detect detect2d.cpp yolo.cpp)

target_link_libraries(detect
    ${OpenCV_LIBS} 
    Threads::Threads 
    zbar
)
