cmake_minimum_required(VERSION 3.11)
project(emotion_detection)

set(CMAKE_CXX_STANDARD 11)

# Include FetchContent module
include(FetchContent)

# Declare the external project
FetchContent_Declare(
  MNN
  GIT_REPOSITORY https://github.com/alibaba/MNN.git
  GIT_TAG master  # or specify a specific tag or branch
)

# Make the project available
FetchContent_MakeAvailable(MNN)

find_package(OpenCV REQUIRED)

include_directories(
  ${mnn_SOURCE_DIR}/include
  src/include
  src
)

add_executable(emotion_detection.out src/main.cpp src/UltraFace/UltraFace.cpp
    src/emotion_detector.cpp)
target_link_libraries(emotion_detection.out MNN ${OpenCV_LIBS})
