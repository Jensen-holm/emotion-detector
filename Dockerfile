# build: docker built -t emotion_detector .
# run: xhost + local:docker
#      docker run --device /dev/video0:/dev/video0 -v $(pwd):/home -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY emotion_detector

FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y \
    libopencv-dev \
    clang \
    cmake
  
WORKDIR /app

COPY . .

RUN cmake . && make

CMD ["./emotion_detection.out"]