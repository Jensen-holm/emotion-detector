cmake_minimum_required(VERSION 3.10)
project(emotion_detection)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(
        src/mnn/include
        src/include
        src
)

link_directories(src/mnn/lib)

add_executable(emotion_detection.out src/main.cpp src/UltraFace/UltraFace.cpp src/emotion_detector.cpp)
target_link_libraries(emotion_detection.out MNN ${OpenCV_LIBS})