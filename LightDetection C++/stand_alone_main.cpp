#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

enum Shape { Circle, Left, Right, Undefined };
typedef std::vector<cv::Point> Contour;

cv::Mat maskImage(cv::Mat& frame, int h, int error, int sMin, int vMin)
{
    cv::Mat hsvImage;
    cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);
    int lowH = (h-error >= 0) ? h-error : h-error+180;
    int highH = (h+error <= 180) ? h+error : h+error-180;

    std::vector<cv::Mat> channels;
    cv::split(hsvImage, channels);
    if (lowH < highH)
        cv::bitwise_and(lowH <= channels[0], channels[0] <= highH, channels[0]);
    else
        cv::bitwise_or(lowH <= channels[0], channels[0] <= highH, channels[0]);
    
    channels[1] = channels[1] > sMin;
    channels[2] = channels[2] > vMin;
    
    cv::Mat mask = channels[0];
    for (int i = 1; i < 3; ++i)
        cv::bitwise_and(channels[i], mask, mask);
    
    cv::Mat grey = cv::Mat::zeros(mask.rows, mask.cols, CV_8U);
    for (int row = 0; row < mask.rows; ++row)
    {
        for (int col = 0; col < mask.cols; ++col)
        {
            auto v1 = channels[0].data[row*mask.cols + col];
            auto v2 = channels[1].data[row*mask.cols + col];
            auto v3 = channels[2].data[row*mask.cols + col];
            grey.data[row*mask.cols + col] = (v1 && v2 && v3 ? 255 : 0);
        }
    }

    return grey;
}

// int calcCCW(cv::Point p1, cv::Point p2, cv::Point p3)
// {
//     return p1.x*p2.y + p2.x*p3.y + p3.x*p1.y - p1.y*p2.x - p2.y*p3.x - p3.y*p1.x;
// }

// bool isConvex(std::vector<cv::Point>& c)
// {
//     if (c.size() < 3)
//         return false;
    
//     bool dir = calcCCW(c[0], c[1], c[2]) >= 0;
//     for (int i = 1; i < (int)c.size(); ++i)
//     {
//         int c1 = i;
//         int c2 = (i + 1) % (int)c.size();
//         int c3 = (i + 2) % (int)c.size();

//         if (dir != (calcCCW(c[c1], c[c2], c[c3]) >= 0))
//             return false;
//     }

//     return true;
// }

bool isConvex(Contour& c, double area)
{
    Contour hull;
    cv::convexHull(c, hull);
    double areaOfHull = cv::moments(hull).m00;
    return abs(areaOfHull - area) / area <= 0.1;
}

Shape labelPolygon(Contour& c, double area)
{
    double peri = cv::arcLength(c, true);
    Contour approx;
    cv::approxPolyDP(c, approx, 0.02*peri, true);

    if ((int)approx.size() == 7)
    {
        cv::Point center = std::accumulate(approx.begin(), approx.end(), cv::Point(0, 0)) / 7;
        int leftCount, rightCount;
        leftCount = rightCount = 0;

        for (int i = 0; i < 7; ++i)
        {
            if ((approx[i] - center).x >= 0)
                ++rightCount;
            else
                ++leftCount;
        }

        if (leftCount > rightCount)
            return Shape::Left;
        else
            return Shape::Right;
    }

    if (approx.size() > 7 && isConvex(c, area))
        return Shape::Circle;
    
    return Shape::Undefined;
}

std::string shapeToString(Shape s)
{
    switch (s)
    {
    case Shape::Circle:
        return "Circle";
    
    case Shape::Left:
        return "Left";

    case Shape::Right:
        return "Right";

    case Shape::Undefined:
        return "Undefined";
    }

    return "Error";
}

std::vector<Contour> findShapes(Shape shapeToFind, cv::Mat& grey, int minArea, int maxArea)
{
    std::vector<Contour> contours;
    cv::findContours(grey, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Contour> found;
    
    for (auto c : contours)
    {
        cv::Moments m;
        m = cv::moments(c);
        
        if (m.m00 != 0 && minArea <= m.m00 && m.m00 <= maxArea)
        {
            Shape shape = labelPolygon(c, m.m00);
            if (shape == shapeToFind)
                found.push_back(c);
        }
    }

    return found;
}

void putTextAtCenter(cv::Mat& frame, std::string text, cv::Scalar color)
{
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseline);
    
    int x = (frame.cols - textSize.width) / 2;
    int y = (frame.rows - textSize.height) / 2;

    cv::putText(frame, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 2, color, 2);
}

int main()
{
    cv::VideoCapture cap(0);
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::namedWindow("Result");

    int minArea, maxArea;
    cv::createTrackbar("Minimum Area", "Result", &minArea, 100000);
    cv::createTrackbar("Maximum Area", "Result", &maxArea, 100000);
    cv::setTrackbarPos("Minimum Area", "Result", 1000);
    cv::setTrackbarPos("Maximum Area", "Result", 100000);

    while (cap.isOpened())
    {
        cv::Mat frame;
        if (cap.read(frame))
        {
            cv::Mat redMasked = maskImage(frame, 0, 15, 180, 128);
            cv::Mat yellowMasked = maskImage(frame, 30, 15, 120, 60);
            cv::Mat greenMasked = maskImage(frame, 60, 15, 90, 60);
            cv::Mat greenInverse = 255 - greenMasked;

            cv::imshow("Found Red", redMasked);
            cv::imshow("Found Yellow", yellowMasked);
            cv::imshow("Found Green", greenMasked);
            cv::imshow("Found Green (Inverse)", greenInverse);

            const static std::string captions[] = { "Red Light!", "Yellow Light!", "Green Light!", "Left Direction!", "Right Direction!" };
            const static cv::Scalar colors[] = { cv::Scalar(0, 0, 255), cv::Scalar(131, 232, 252), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0) };

            std::vector<Contour> found[] = {
                findShapes(Shape::Circle, redMasked, minArea, maxArea),
                findShapes(Shape::Circle, yellowMasked, minArea, maxArea),
                findShapes(Shape::Circle, greenMasked, minArea, maxArea),
                findShapes(Shape::Left, greenInverse, minArea, maxArea),
                findShapes(Shape::Right, greenInverse, minArea, maxArea)
            };

            for (int i = 0; i < 5; ++i)
            {
                if (!found[i].empty())
                {
                    cv::drawContours(frame, found[i], -1, colors[i], 2);
                    putTextAtCenter(frame, captions[i], colors[i]);
                }
            }

            cv::imshow("Result", frame);
        }

        int key = cv::waitKey(1) & 0xff;
        if (key == 27)
            break;
    }

    return 0;
}