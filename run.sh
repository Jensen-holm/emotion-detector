#! /bin/bash

xhost + local:docker

docker run --device /dev/video0:/dev/video0 -v $(pwd):/home -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY emotion_detector
