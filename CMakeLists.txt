cmake_minimum_required(VERSION 3.9)
project(fake_smile)

set(CMAKE_CXX_STANDARD 11)

add_executable(fake_smile main.cpp)

find_package( OpenCV REQUIRED )

target_link_libraries(fake_smile ${OpenCV_LIBS})

find_package(dlib REQUIRED)

include(/usr/local/lib/cmake/dlib/dlib.cmake)

target_link_libraries(fake_smile dlib::dlib)