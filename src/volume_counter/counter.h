#pragma once


#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"

#include "opencv2/highgui/highgui.hpp"
#include <iostream> 
#include <stdio.h>
#include <math.h>

#define ROUND(x) (static_cast<int>(x + 0.5))

struct objectType
{
    double _width, _height, _depth;
    std::string _name;
    objectType(double width, double height, double depth, std::string name)
    {
        _width = width;
        _height = height;
        _depth = depth;
        _name = name;
    }
};

class counter
{
    private:
        
        cv::Mat _src; // the source image
        cv::Point3d _separator; // the point where we start to segment the specific line of products
        objectType _object; // the type of the object being counted
        cv::Mat _rotation = cv::Mat::zeros(3, 3, CV_64FC1);

    public:
        counter(): _object(0,0,0,""){}
        ~counter(){}
        
        void setImg(cv::Mat& img)
        {
            _src = img;
        }
        
        void setSeparator(cv::Point3d& separator)
        {
            _separator = separator;
        }

        void setType(double width, double height, double depth, std::string name)
        {
            _object._width = width;
            _object._height = height;
            _object._depth = depth;
            _object._name = name;
        }
        
        void setRotation(cv::Mat& rotation)
        {
            _rotation = rotation;
        }

        objectType& getObject()
        {
            return _object;
        }

        double countObjects();
};
