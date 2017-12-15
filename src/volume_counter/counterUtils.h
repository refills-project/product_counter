/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.
*******************************************************************************/
#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 4800) // for allowing sliders on bool variable w/o warning
#endif

#include <string>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui//highgui.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <numeric>

namespace counterUtils
{
// Returns a random number (uniform distr.)  in  [0, len-1]
unsigned int randIdx(unsigned int len);

// from http://www.rapidtables.com/web/color/RGB_Color.htm
const std::vector<cv::Scalar> stdColors{
    cv::Scalar(0, 0, 0),
    cv::Scalar(255, 255, 255),
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 255),
    cv::Scalar(192, 192, 192),
    cv::Scalar(128, 128, 128),
    cv::Scalar(128, 0, 0),
    cv::Scalar(128, 128, 0),
    cv::Scalar(0, 128, 0),
    cv::Scalar(128, 0, 128),
    cv::Scalar(0, 128, 128),
    cv::Scalar(0, 0, 128),

};
cv::Scalar randColor();

// Alternately display two images (e.g. aligned depth & color), blink effect.
void dynamicImageComparison(const cv::Mat &img1, const cv::Mat &img2, double resizeFactor = 1.0, int blinkTime = 1000);
// Improvement of cv::putText for enhanced visibility.
void outlinedText(cv::Mat &img, const std::string &text, cv::Point org,
                  int fontFace, double fontScale, cv::Scalar fgColor, cv::Scalar bgColor,
                  int thickness = 1, int lineType = 8, bool bottomLeftOrigin = false);

// Wrapping of OCV canny
cv::Mat cannyEdgeDetection(const cv::Mat &src, int lowThreshold = 20);

cv::Mat drawTwoEdgeMaps(const cv::Mat &redEdges, const cv::Mat &greenEdges);

cv::Mat drawEdgeCorrespondences(const cv::Mat &sSrc, const cv::Mat &sTarget,
                                const cv::Mat &correspondences, bool displayIt = true);

// Wrapper of openCV trackbar, allowing attaching variable of any standard type to a trackbar.
// The supported variable range is on the interval [0, maxVal * scaler]. Zero can be excluded.
// Usage: full constructor at init, then update() in the main loop.
// The additional methods are for conveniency only.
template <class varType>
class OCVTrackbar
{
  public:
    OCVTrackbar(const std::string &barName, const std::string &winName, varType *pVar, varType maxVal, float scaler = 1.f, bool allowZero = true)
        : m_winName(winName), m_barName(barName)
    {
        attach(pVar, maxVal, scaler, allowZero);
    }
    OCVTrackbar() : m_winName("win"), m_barName("bar"), m_pVar(nullptr)
    {
    }

    void setNames(const std::string &barName, const std::string &winName)
    {
        m_winName = winName;
        m_barName = barName;
    }
    virtual void attach(varType *pVar, varType maxVal, float scaler, bool allowZero = true)
    {
        m_pVar = pVar;
        m_scaler = scaler;
        m_maxValUI = (int)(maxVal * m_scaler);
        m_varUI = (int)(*m_pVar * m_scaler);
        m_allowZero = allowZero;

        m_prevVal = m_varUI;
    }
    virtual ~OCVTrackbar(){};

    bool update() // returns true if the value has been modified
    {
        if (m_pVar == nullptr)
        {
            std::cout << "Error: Trackbar " << m_barName << " was not initialized ('attach' function)." << std::endl;
            return false;
        }

        cv::createTrackbar(m_barName, m_winName, &m_varUI, m_maxValUI);

        if (!(m_allowZero == false && m_varUI == 0))
        {
            *m_pVar = static_cast<varType>(m_varUI / m_scaler);
        }
        else
        {
            setValue(*m_pVar); // zero is not allowed, put slider back to last value.
        }

        bool hasChanged = (m_varUI != m_prevVal);
        if (hasChanged)
        {
            m_prevVal = m_varUI;
        }
        return hasChanged;
    }
    varType getValue() const
    {
        return *m_pVar;
    }
    void setValue(const varType &val)
    {
        //return;
        *m_pVar = val;
        m_prevVal = m_varUI = (int)(*m_pVar * m_scaler);
        if (m_varUI <= m_maxValUI)
        {
            cv::setTrackbarPos(m_barName, m_winName, m_varUI); //TODO: SEGFAULT here : check
        }
    }

  protected:
    std::string m_winName, m_barName;
    varType *m_pVar;
    int m_varUI;
    float m_scaler;
    int m_maxValUI;
    bool m_allowZero;

    int m_prevVal;
};

//cv::Mat getColorCodedDepth(const cv::Mat &depth, int minz, int maxz);
//cv::Mat displayColorCodedDepth(const std::string &winName, const cv::Mat &depth);
}

//////////////////////////////
// Basic Data Types         //
//////////////////////////////

struct float3
{
    float x, y, z;
};
struct float2
{
    float x, y;
};

struct rect
{
    float x, y;
    float w, h;

    // Create new rect within original boundaries with give aspect ration
    rect adjust_ratio(float2 size) const
    {
        auto H = static_cast<float>(h), W = static_cast<float>(h) * size.x / size.y;
        if (W > w)
        {
            auto scale = w / W;
            W *= scale;
            H *= scale;
        }

        return {x + (w - W) / 2, y + (h - H) / 2, W, H};
    }
};

/////////////////////////////////////////////
////////////////////////
// Image display code //
////////////////////////

//inline void draw_text(int x, int y, const char * text)
//{
//    char buffer[60000]; // ~300 chars
//    glEnableClientState(GL_VERTEX_ARRAY);
//    glVertexPointer(2, GL_FLOAT, 16, buffer);
//    glDrawArrays(GL_QUADS, 0, 4 * stb_easy_font_print((float)x, (float)(y - 7), (char *)text, nullptr, buffer, sizeof(buffer)));
//    glDisableClientState(GL_VERTEX_ARRAY);
//}

//class texture
//{
//  public:
//    void render(const rs2::video_frame &frame, const rect &r)
//    {
//        upload(frame);
//        show(r.adjust_ratio({float(width), float(height)}));
//    }

//    void upload(const rs2::video_frame &frame)
//    {
//        if (!frame)
//            return;
//        if (!gl_handle)
//            glGenTextures(1, &gl_handle);
//        GLenum err = glGetError();
//        auto format = frame.get_profile().format();
//        width = frame.get_width();
//        height = frame.get_height();
//        stream = frame.get_profile().stream_type();

//        glBindTexture(GL_TEXTURE_2D, gl_handle);

//        switch (format)
//        {
//        case RS2_FORMAT_RGB8:
//            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.get_data());
//            break;
//        default:
//            throw std::runtime_error("The requested format is not suported by this demo!");
//        }

//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
//        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
//        glBindTexture(GL_TEXTURE_2D, 0);
//    }

//    GLuint get_gl_handle() { return gl_handle; }

//    void show(const rect &r) const
//    {
//        if (!gl_handle)
//            return;

//        glBindTexture(GL_TEXTURE_2D, gl_handle);
//        glEnable(GL_TEXTURE_2D);
//        glBegin(GL_QUAD_STRIP);
//        glTexCoord2f(0.f, 1.f);
//        glVertex2f(r.x, r.y + r.h);
//        glTexCoord2f(0.f, 0.f);
//        glVertex2f(r.x, r.y);
//        glTexCoord2f(1.f, 1.f);
//        glVertex2f(r.x + r.w, r.y + r.h);
//        glTexCoord2f(1.f, 0.f);
//        glVertex2f(r.x + r.w, r.y);
//        glEnd();
//        glDisable(GL_TEXTURE_2D);
//        glBindTexture(GL_TEXTURE_2D, 0);

//        draw_text(r.x + 15, r.y + 20, rs2_stream_to_string(stream));
//    }

//  private:
//    GLuint gl_handle = 0;
//    int width = 0;
//    int height = 0;
//    rs2_stream stream = RS2_STREAM_ANY;
//};

//class window
//{
//  public:
//    std::function<void(bool)> on_left_mouse = [](bool) {};
//    std::function<void(double, double)> on_mouse_scroll = [](double, double) {};
//    std::function<void(double, double)> on_mouse_move = [](double, double) {};
//    std::function<void(int)> on_key_release = [](int) {};

//    window(int width, int height, const char *title)
//        : _width(width), _height(height)
//    {
//        glfwInit();
//        win = glfwCreateWindow(width, height, title, nullptr, nullptr);
//        glfwMakeContextCurrent(win);

//        glfwSetWindowUserPointer(win, this);
//        glfwSetMouseButtonCallback(win, [](GLFWwindow *win, int button, int action, int mods) {
//            auto s = (window *)glfwGetWindowUserPointer(win);
//            if (button == 0)
//                s->on_left_mouse(action == GLFW_PRESS);
//        });

//        glfwSetScrollCallback(win, [](GLFWwindow *win, double xoffset, double yoffset) {
//            auto s = (window *)glfwGetWindowUserPointer(win);
//            s->on_mouse_scroll(xoffset, yoffset);
//        });

//        glfwSetCursorPosCallback(win, [](GLFWwindow *win, double x, double y) {
//            auto s = (window *)glfwGetWindowUserPointer(win);
//            s->on_mouse_move(x, y);
//        });

//        glfwSetKeyCallback(win, [](GLFWwindow *win, int key, int scancode, int action, int mods) {
//            auto s = (window *)glfwGetWindowUserPointer(win);
//            if (0 == action) // on key release
//            {
//                s->on_key_release(key);
//            }
//        });
//    }

//    float width() const { return float(_width); }
//    float height() const { return float(_height); }

//    operator bool()
//    {
//        glPopMatrix();
//        glfwSwapBuffers(win);

//        auto res = !glfwWindowShouldClose(win);

//        glfwPollEvents();
//        glfwGetFramebufferSize(win, &_width, &_height);

//        // Clear the framebuffer
//        glClear(GL_COLOR_BUFFER_BIT);
//        glViewport(0, 0, _width, _height);

//        // Draw the images
//        glPushMatrix();
//        glfwGetWindowSize(win, &_width, &_height);
//        glOrtho(0, _width, _height, 0, -1, +1);

//        return res;
//    }

//    ~window()
//    {
//        glfwDestroyWindow(win);
//        glfwTerminate();
//    }

//    operator GLFWwindow *() { return win; }

//  private:
//    GLFWwindow *win;
//    int _width, _height;
//};

namespace EPV
{
#define ZERO_EPSILON (1.0e-10)
	// Types__________________________________
	struct CameraIntrinsics
	{
		double fx;
		double fy;
		double px;
		double py;
		uint32_t w;
		uint32_t h;
		double k[5]; // not used
	};

	struct RansacParams
	{
		unsigned int        numIterations;
		double              percentilePoint;
		double              inlierThreshFactor;
		double              nnz2MinInliers;
		bool                refinePlane;

		RansacParams()
		{
			numIterations = 1000; //  100-200 is generally sufficient and much faster
			percentilePoint = 0.2;
			inlierThreshFactor = 1.0;   // this is a multiplier for the empiric percentile2inlier function of z. (see below)
			nnz2MinInliers = 0.15;
			refinePlane = false; // recommended to set false
		}
	};

	typedef std::array<double, 4>   planeParams; // Plane Coefs s.t. a point (x, y, z) on this plane verifies: a[0]*x + a[1]*y +a[2]*z + a[3] = 0

	struct FittedPlane
	{
		planeParams                 planeEquation;
		std::vector<cv::Point3d>    inliers;
		cv::Mat                     inliersMask;
		double                      inlierThreshold;
		double                      score;

		FittedPlane() :inlierThreshold(0.0), score(0.0) {}
	};

	struct DepthImageCoords
	{
		double x, y, z, u, v;

		cv::Point3d     worldCoords()
		{
			return cv::Point3d(x, y, z);
		}
		cv::Point       imageCoords()
		{
			return cv::Point((int)u, (int)v);
		}
	};
	//__________________________________________________________________________________________________
	void iterativePlaneFitting(const cv::Mat& depthImg, const CameraIntrinsics& intrinsics,
		const cv::Rect& ROI, std::vector<FittedPlane>& results, const RansacParams& params, std::vector<EPV::DepthImageCoords>& nnzCoords);

	//__________________________________________________________________________________________________
	void ransacPlaneFitting(const cv::Mat& img, const CameraIntrinsics& intrinsics, const cv::Point& topLeftOffset, FittedPlane& res,
		const RansacParams& params, std::vector<DepthImageCoords>& nnzCoords);
	
	//__________________________________________________________________________________________________
	double computePercentilePoint(const cv::Mat& img, double percentile, const float range[]);
        void depthImageToWorldCoord_depth(const cv::Mat& img, const CameraIntrinsics& intrinsics, std::vector<DepthImageCoords>& coords,
		const cv::Point offset);
} // namespace EPV

//--------------------------------------------------------------------------------------

namespace TransformUtils
{
	const double RAD2DEG = 180 / CV_PI;
	const double DEG2RAD = CV_PI / 180;
	// Useful conversions ___________________________________________________________________
	cv::Mat axisAngleToRotMat(double x, double y, double z, double angle);
	//--------------------------------------------------------------------------------------
	cv::Mat eulerXYZToRotMat(double x, double y, double z);
	//--------------------------------------------------------------------------------------
	// Input: quat is 1x4 or 4x1 CV_64F
	cv::Mat quaternionToRotMat(const cv::Mat& quat);
	cv::Mat quaternionToRotMat(double x, double y, double z, double w);
    //--------------------------------------------------------------------------------------
	void rotMatToEulerXYZ(cv::Mat R, double& x, double& y, double& z, bool inDegrees = false);
	//--------------------------------------------------------------------------------------
	// convert a homogeneous coordinates transf matrix [R|t] of size 3x4 to a 4x4 matrix
	cv::Mat homogenize34(const cv::Mat& m);
	// Rigid body transformations __________________________________________________________
	// Input: data is Nx3, R 3x3, t is 3x1 (or Rt is 3x4)
	cv::Mat doTransform(const cv::Mat& data, const cv::Mat& R, const cv::Mat& t);
	cv::Mat doTransform(const cv::Mat& data, const cv::Mat& Rt);
	cv::Mat doInverseTransform(const cv::Mat& data, const cv::Mat& R, const cv::Mat& t);
	cv::Mat doInverseTransform(const cv::Mat& data, const cv::Mat& Rt);
	//--------------------------------------------------------------------------------------
	// Compute body local axes, given 3 (unaligned) points
	cv::Mat computeAxes(const cv::Mat& ABC);
	//--------------------------------------------------------------------------------------
	// computeBodyTransform from XYZ1 to XYZ2
	double computeBodyTransform(const cv::Mat& XYZ1, const cv::Mat& XYZ2, cv::Mat& R, cv::Mat& t);
	// Evaluation  __________________________________________________________
	double computeAvgDistance(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& dist);
	//--------------------------------------------------------------------------------------
	void matchPointSets(const cv::Mat XYZ1, const cv::Mat XYZ2, float maxDistance, cv::Mat& res1, cv::Mat& res2);
	// 3D Clouds  __________________________________________________________
	cv::Mat createTriadCloud();
	// convert a CV64FC1 Nx3 cloud (in meters) into a CV_16UC1 depth map (in mm).
	cv::Mat projectCloud(const cv::Mat& cloud, unsigned int w, unsigned int h, double fx, double fy, double cx, double cy, bool inMM = true);
	// convert a CV64FC1 Nx3 cloud (in meters) with CV_8UC1 Nx3 RGB info into a CV_8UC3 map.
	cv::Mat projectColoredCloud(const cv::Mat& cloud, const cv::Mat& rgb, unsigned int w, unsigned int h, double fx, double fy, double cx, double cy, bool inMM = true);
	// convert a a CV_16UC1 depth map (in mm) into a CV64FC1 Nx3 cloud (in meters).
	cv::Mat unprojectCloud(const cv::Mat& depth, double fx, double fy, double cx, double cy, int minz = 200, int maxz = 10000, bool inMM = true);
	cv::Mat unprojectColoredCloud(const cv::Mat& depth, const cv::Mat& rgb, double fx, double fy, double cx, double cy, int minz = 200, int maxz = 10000, bool inMM = true);
}
//--------------------------------------------------------------------------------------
