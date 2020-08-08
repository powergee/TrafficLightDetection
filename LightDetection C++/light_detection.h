#ifndef LIGHT_DETECTION_H_
#define LIGHT_DETECTION_H_

#include <opencv2/opencv.hpp>

struct TrafficLights
{
    bool red, yellow, green, left, right;
};

enum Shape {
    Circle, Left, Right, Undefined
};
typedef std::vector<cv::Point> Contour;

cv::Mat maskImage(cv::Mat& frame, int h, int error, int sMin, int vMin);
bool isConvex(Contour& c, double area);
Shape labelPolygon(Contour& c, double area);
std::string shapeToString(Shape s);
std::vector<Contour> findShapes(Shape shapeToFind, cv::Mat& grey, int minArea, int maxArea);
void putTextAtCenter(cv::Mat& frame, std::string text, cv::Scalar color);
TrafficLights detectAll(cv::Mat& frame, bool drawOnFrame = false, int minArea = 1000, int maxArea = 100000);

#endif