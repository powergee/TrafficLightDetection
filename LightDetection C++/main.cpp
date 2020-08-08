#include <iostream>
#include "light_detection.h"

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
    std::cout << std::boolalpha;

    while (cap.isOpened())
    {
        cv::Mat frame;
        if (cap.read(frame))
        {
            TrafficLights lights = detectAll(frame, false, minArea, maxArea);
            std::cout << lights.red << " ";
            std::cout << lights.yellow << " ";
            std::cout << lights.green << " ";
            std::cout << lights.left << " ";
            std::cout << lights.right << "\n";

            cv::imshow("Result", frame);
        }

        int key = cv::waitKey(1) & 0xff;
        if (key == 27)
            break;
    }

    return 0;
}