FROM thecanadianroot/opencv-cuda

WORKDIR /app

COPY . .

RUN cmake . && make

CMD [ "./emotion_detection.out" ]
