#ifndef EMOTIONS_HPP
#define EMOTIONS_HPP
#include <map>
#include <opencv2/opencv.hpp>
#include <string>

const std::string ANGRY_EMOJI_PATH = "assets/emojis/angry_emoji.png";
const std::string SAD_EMOJI_PATH = "assets/emojis/sad_emoji.png";
const std::string NEUTRAL_EMOJI_PATH = "assets/emojis/neutral_emoji.png";
const std::string FEAR_EMOJI_PATH = "assets/emojis/scared_emoji.png";
const std::string HAPPY_EMOJI_PATH = "assets/emojis/smiling_emoji.png";
const std::string SURPRISED_EMOJI_PATH = "assets/emojis/surprised_emoji.png";
const std::string DISGUST_EMOJI_PATH = "assets/emojis/disgust_emoji.png";

std::map<int, cv::Mat> loadEmojiMap() {
  return {
      {0, cv::imread(ANGRY_EMOJI_PATH)},
      {1, cv::imread(DISGUST_EMOJI_PATH)},
      {2, cv::imread(FEAR_EMOJI_PATH)},
      {3, cv::imread(HAPPY_EMOJI_PATH)},
      {4, cv::imread(SAD_EMOJI_PATH)},
      {5, cv::imread(SURPRISED_EMOJI_PATH)},
      {6, cv::imread(NEUTRAL_EMOJI_PATH)},
  };
};

#endif // EMOTIONS_HPP
