# build: docker built -t emotion_detector .
# run: xhost + local:docker
#      docker run --device /dev/video0:/dev/video0 -v $(pwd):/home -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY emotion_detector

# Use the lightweight Alpine base image
FROM alpine:latest

# Install necessary packages
RUN apk add --no-cache \
    opencv-dev \
    clang \
    cmake \
    make \
    g++ \
    git \
    linux-headers

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Build the application
RUN mkdir build && cd build && cmake .. && make

# Define the entry point with the correct path
ENTRYPOINT ["./build/emotion_detection.out"]
