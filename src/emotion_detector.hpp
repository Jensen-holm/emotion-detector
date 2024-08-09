#ifndef EMOTION_DETECTOR_HPP
#define EMOTION_DETECTOR_HPP

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

const std::string EMOTION_DETECTOR_PATH = "model/emotion_detection.pb";

class EmotionDetector {
private:
  cv::dnn::Net model;

  void preProcessInput(cv::Mat *croppedFace, cv::Mat *dest);

public:
  EmotionDetector() {
    model = cv::dnn::readNetFromTensorflow(EMOTION_DETECTOR_PATH);
  };

  int predict(cv::Mat *croppedFace);
};

#endif // EMOTION_DETECTOR_HPP