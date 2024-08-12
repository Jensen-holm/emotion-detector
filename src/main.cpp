#include "UltraFace/UltraFace.hpp"
#include "emotion_detector.hpp"
#include "emotions.hpp"

#include <iostream>
#include <string>
#include <map>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


const std::string FACE_DETECTOR_PATH = "model/slim-320.mnn";
const std::string WINDOW_NAME = "Emotion Detector";


void overlayFace(cv::Mat croppedFace, cv::Mat origFrame, cv::Mat emotionImg, FaceInfo face) {
  cv::Mat resizedEmotionImg;
  cv::resize(emotionImg, resizedEmotionImg,
              cv::Size(croppedFace.cols, croppedFace.rows),
              cv::INTER_LINEAR);
  
  cv::Mat zeroMask = (resizedEmotionImg == 0);
  cv::Mat flippedEmoji = cv::Mat::ones(zeroMask.size(), zeroMask.type());
  flippedEmoji.setTo(1, zeroMask);

  cv::Mat mulResult;
  cv::multiply(flippedEmoji, croppedFace, mulResult);
  cv::add(resizedEmotionImg, mulResult, resizedEmotionImg);

  cv::Mat insetImage(origFrame, cv::Rect(face.x1, face.y1, croppedFace.cols,
                                      croppedFace.rows));
  resizedEmotionImg.copyTo(insetImage);
}

int main(int argc, char *argv[]) {
  int windowHeight;
  int windowWidth;
  if (argc < 3) {
    windowHeight = 1080;
    windowWidth = 1920;
  } else {
    windowWidth = atol(argv[1]);
    windowHeight = atol(argv[2]);
  }

  // load face detector model and emotion detector model
  UltraFace ultraface(FACE_DETECTOR_PATH, 320, 240, 4, 0.65);
  EmotionDetector emotionDetector;

  // load emoji map
  std::map<int, cv::Mat> emojiMap = loadEmojiMap();

  // open video capture & window
  cv::VideoCapture cap(0);
  std::cout << "Running emotion detector with " << windowWidth << "x"
            << windowHeight << " resolution." << std::endl;
  if (!cap.isOpened()) {
    std::cout << "VideoCapture was not opened." << std::endl;
    return -1;
  }

  cv::namedWindow(WINDOW_NAME, cv::WND_PROP_FULLSCREEN);
  while (1) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      std::cout << "empty frame" << std::endl;
      break;
    }

    std::vector<FaceInfo> faceInfo;
    ultraface.detect(frame, faceInfo);
    for (auto face : faceInfo) {
      // crop the face out of the frame
      cv::Mat croppedFace =
          frame(cv::Range(face.y1, face.y2), cv::Range(face.x1, face.x2));

      // predict the emotion of the face
      int emotion = emotionDetector.predict(&croppedFace);

      // get the corresponding emoji from the emoji map
      cv::Mat emotionImg = emojiMap.at(emotion);
      overlayFace(croppedFace, frame, emotionImg, face);
    }

    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, cv::Size(windowWidth, windowHeight),
               cv::INTER_LINEAR);

    cv::imshow(WINDOW_NAME, resizedFrame);
    if (cv::waitKey(30) == 27) {
      break; // hit ESC to quit
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
