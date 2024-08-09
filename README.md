# Real Time Multiple Face Emotion Detection

| | | |
|:---------------------:|:-------------------:|:-----------------:|
|<img width="500" alt="happy example" src="assets/example_output/happy_example.png"> | <img width="500" alt="angry example" src="assets/example_output/angry_example.png"> | <img width="500" alt="sad example" src="assets/example_output/sad_example.png">|
| <img width="500" alt="surprised example" src="assets/example_output/surprised_example.png"> | <img width="500" alt="neutral example" src="assets/example_output/neutral_example.png"> | <img width="500" alt="fear example" src="assets/example_output/fear_example.png"> |
| | | |


## Face Detection

I decided to use a face detection model that I found in [this](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) github repository. The code was almost directly pulled from the MNN directory. I used the package that the author wrote (UltraFace) to extract the faces in images from opencv captured images.


## Emotion Prediction

The model we are using for this project was found on github [here](https://github.com/martycheung/CppND-Facial-Emotion-Recognition/blob/master/model/Facial_Emotion_Recognition_Model_CNN.ipynb).

The model we are using to predict emotion was trained on this [kaggle dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=train.csv) 


## Getting Started

**Dependencies**
- opencv

**Build & Run** <br>
1. `cmake .` <br>
2. `make` <br>
3. `./emotion_detection.out` <br>

**Docker useage**
1. `docker build -t emotion_detector .` <br>
2. `xhost + local:docker` <br>
3. `docker run --device /dev/video0:/dev/video0 -v $(pwd):/home -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY emotion_detector` <br>
