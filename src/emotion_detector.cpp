#include "emotion_detector.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/**
 * @brief converts an input face image into a blob that our model can understand 
 * 
 * @param croppedFace bgr colorspace image of a face 
 * @param dest destination matrix for the processed input
 * @return void 
 */
void EmotionDetector::preProcessInput(cv::Mat* croppedFace, cv::Mat* dest) {
  cv::Mat grayCroppedFace;
  cv::cvtColor(*croppedFace, grayCroppedFace, cv::COLOR_BGR2GRAY);
  cv::dnn::blobFromImage(grayCroppedFace, *dest, 1.0 / 255.0, cv::Size(48, 48), cv::Scalar(0, 0, 0), false);
}

/**
 * @brief predict the emotion of a person using the opencv dnn model 
 * 
 * @param croppedFace pointer to opencv matrix object in bgr colorspace that is just the face 
 * @return int 0-6, 0=anger, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprised, 6=neutral 
 */
int EmotionDetector::predict(cv::Mat* croppedFace) {
  cv::Mat blob;
  preProcessInput(croppedFace, &blob);
  model.setInput(blob);
  cv::Mat pred = model.forward();

  double min, max;
  cv::Point minLoc, maxLoc;
  cv::minMaxLoc(pred, &min, &max, &minLoc, &maxLoc);
  return maxLoc.x;
}
