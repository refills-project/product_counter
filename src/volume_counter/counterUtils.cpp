/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#include "counterUtils.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <random>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

//...................................................................................................
namespace counterUtils
{
//...................................................................................................
unsigned int randIdx(unsigned int len)
{
    //from http://www.cppsamples.com/common-tasks/choose-random-element.html
    static std::random_device random_device;
    static std::mt19937 engine{ random_device() };
    static std::uniform_int_distribution<int> dist(0, len - 1);
    return dist(engine);
}

cv::Scalar randColor()
{
    return stdColors[randIdx((unsigned int)stdColors.size())];
}


//...................................................................................................
cv::Mat cannyEdgeDetection(const cv::Mat& src, int lowThreshold)
{
    cv::Mat src_gray;
    cv::Mat dst, detected_edges;

    int edgeThresh = 1;
    int ratio = 3;
    int kernel_size = 3;
    char* window_name = "Edge Map";

    if (src.channels() > 1)
    {
        cv::cvtColor(src, src_gray, CV_BGR2GRAY);
    }
    else
    {
        src_gray = src.clone();
    }
    /// Reduce noise with a kernel 3x3
    cv::blur(src_gray, src_gray, cv::Size(3, 3));

    /// Canny detector
    cv::Canny(src_gray, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

    return detected_edges;
};

//...................................................................................................
cv::Mat drawTwoEdgeMaps(const cv::Mat& redEdges, const cv::Mat& greenEdges)
{
    assert(redEdges.size() == greenEdges.size());
    cv::Mat disp(redEdges.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    disp.setTo(cv::Scalar(0, 0, 255), redEdges);
    disp.setTo(cv::Scalar(0, 255, 0), greenEdges);
    return disp;
}
//...................................................................................................
//cv::Mat drawEdgeCorrespondences(const cv::Mat& sSrc, const cv::Mat& sTarget, const cv::Mat& correspondences, bool displayIt)
//{
//    // draw
//    cv::Mat disp = drawTwoEdgeMaps(sSrc, sTarget);
//    cv::Mat disp0 = disp.clone();
//    for (int i = 0; i < correspondences.rows; i += 10)
//    {
//        for (int j = 0; j < correspondences.cols; ++j)
//        {
//            cv::Point p2 = (correspondences.type() == CV_32FC2) ?
//                           (cv::Point)correspondences.at<cv::Point2f>(i, j):
//                           correspondences.at<cv::Point>(i, j);
//            if (p2.x + p2.y > 0)
//            {
//                cv::Point p1 = cv::Point(j, i);
//                cv::Scalar color = counterUtils::randColor();

//                cv::line(disp, p1, p2, color);
//                cv::circle(disp, p1, 1, color);
//                cv::circle(disp, p2, 1, color);
//            }
//        }
//    }
//    if (displayIt)
//    {
//        counterUtils::dynamicImageComparison(disp0, disp, 1.0, 500);
//    }
//    return disp;

//}
//__________________________________________________________________________________________________
// Alternately display two images. Both image sizes are set as the first's, times resizeFactor.
void dynamicImageComparison(const cv::Mat& img1, const cv::Mat& img2, double resizeFactor, int blinkTime)
{
    cv::Mat img[2] = { img1.clone(), img2.clone() };
    if (resizeFactor != 1.0)
    {
        cv::resize(img[0], img[0], cv::Size(), resizeFactor, resizeFactor);
    }
    if (img[1].size() != img[0].size())
    {
        cv::resize(img[1], img[1], img[0].size());
    }
    std::string winName = "Image comparison";
    cv::namedWindow(winName);
    int currentlyDisplaying = 0;
    int key = 0;
    while (key != 27)
    {
        imshow(winName, img[currentlyDisplaying]);
        key = cv::waitKey(blinkTime);
        currentlyDisplaying = 1 - currentlyDisplaying;
    }
    cv::destroyWindow(winName);
}

//__________________________________________________________________________________________________
void outlinedText(cv::Mat& img, const std::string& text, cv::Point org,
                  int fontFace, double fontScale, cv::Scalar fgColor, cv::Scalar bgColor,
                  int thickness, int lineType, bool bottomLeftOrigin)
{
    cv::putText(img, text, org, fontFace, fontScale, bgColor, thickness * 3, lineType, bottomLeftOrigin);
    cv::putText(img, text, org, fontFace, fontScale, fgColor, thickness, lineType, bottomLeftOrigin);
}

//__________________________________________________________________________________________________
//cv::Mat getColorCodedDepth(const cv::Mat& depth, int minz, int maxz)
//{
//	cv::Mat depth_disp;
//	depth.convertTo(depth_disp, CV_32FC1);
//	depth_disp.setTo(minz, depth_disp < minz);
//	depth_disp.setTo(maxz, depth_disp > maxz);
//	float depth_scale_factor = 1.0f / (maxz - minz);

//	depth_disp -= minz;
//	depth_disp *= depth_scale_factor;
//	cv::log(1 + depth_disp, depth_disp);

//	depth_disp *= (255 / cv::log(2));
//	depth_disp.convertTo(depth_disp, CV_8UC1);

//	cv::Mat cmap;
//	cv::applyColorMap(depth_disp, cmap, cv::COLORMAP_JET);
//#if 1
//	cmap.setTo(cv::Scalar(255, 255, 255), depth < minz);
//	cmap.setTo(cv::Scalar(100, 100, 100), depth == 0);
//	cmap.setTo(cv::Scalar(0, 0, 0), depth >= maxz);
//#endif
//	return cmap;

//}

//__________________________________________________________________________________________________

//cv::Mat displayColorCodedDepth(const std::string& winName, const cv::Mat& depth)
//{
//	static int minz = 200, maxz = 4000; // static for being consistent between all windows
//	cv::Mat depthCpy;
//	if (depth.type() != CV_16UC1)
//	{
//		depth.convertTo(depthCpy, CV_16UC1);
//	}
//	else
//	{
//		depthCpy = depth;
//	}
//	cv::Mat cmap = getColorCodedDepth(depthCpy, minz, maxz);

//	cv::imshow(winName, cmap);
//	cv::createTrackbar("minz", winName, &minz, 1500);
//	cv::createTrackbar("maxz", winName, &maxz, 5000);
//	if (maxz<minz)
//	{
//		minz = maxz - 50;
//		if (minz < 0)
//		{
//			minz = 0;
//			maxz = 50;
//		}
//	}
//	return cmap;
//}

}

namespace EPV
{
	//__________________________________________________________________________________________________
	static double percentile2InlierThresh(double z)
	{
		return std::max(0.5, (4E-06*z*z + 0.0169*z - 5.6805)); // z in mm, empirical DS4
	}

	//__________________________________________________________________________________________________
	void iterativePlaneFitting(const cv::Mat& depthImg, const CameraIntrinsics& intrinsics,
		const cv::Rect& ROI, std::vector<FittedPlane>& results, const RansacParams& params, std::vector<EPV::DepthImageCoords>& nnzCoords)
	{
		results.clear();
		cv::Mat imgClone = depthImg(ROI).clone();
		// Local parameters------------------------------------
		// bool     biggestCCOnly = false;
		// NOTE TO HELING: The result of Ransac can be composed of several patches (e.g. the motorcycle in the ppt slide 25)
		// It might be relevant to you to use connected components over the inlier mask for restricting each plane to be connected


		const uint32_t  minNonZeros = 1000;
		const uint32_t  minInliers = static_cast<uint32_t> (cv::countNonZero(imgClone > ZERO_EPSILON) * params.nnz2MinInliers);  //  = nonZeros / 20
		const uint32_t  failLimit = 5;      //  number of allowed successive failures

		uint32_t    failCounter = 0;
		FittedPlane res;
		bool splane = false, wplane = false;

		// Apply ransac to imageClone, remove the plane inliers from imageClone, and so on
		while ((uint32_t)cv::countNonZero(imgClone > ZERO_EPSILON) >= minNonZeros)
		{
			// Run ransacPlaneFitting
			FittedPlane fittedPlane;
			ransacPlaneFitting(imgClone, intrinsics, ROI.tl(), fittedPlane, params, nnzCoords);

			// Manage failure
			if (fittedPlane.inliers.size() < minInliers)
			{
				if (failCounter++ < failLimit)
				{
					continue;
				}
				else
				{
					break;
				}
			}

			// Found a plane!

			//TODO (if ever...): extract the biggest connected component only.
			// Alternative (potentially more accurate but slower, to check):
			// tweak ransacPlaneFitting to return the CONNECTED plane with the highest number of inliers
			imgClone = imgClone.setTo(0.f, fittedPlane.inliersMask);
			if (fittedPlane.planeEquation[3] < 600) // the shelf's normal
			{
				if (fabs(fittedPlane.planeEquation[1]) > fabs(fittedPlane.planeEquation[2]) && fabs(fittedPlane.planeEquation[1]) > fabs(fittedPlane.planeEquation[0]) && !splane)
				{
					results.emplace_back(fittedPlane);
					splane = true;
				}
			}
			else if(fittedPlane.planeEquation[3] >= 600 && !wplane)
			{
				results.emplace_back(fittedPlane);
				wplane = true;
			}
			if(wplane && splane)
				break;
			failCounter = 0;
		}
	}
	//__________________________________________________________________________________________________
	double computePercentilePoint(const cv::Mat& img, double percentile, const float range[])
	{
		/// Compute  histogram
		int histSize = 1024;
		const float* histRange[] = { range };
		bool uniform = true;
		bool accumulate = false;
		cv::Mat hist;
		cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate);
		// Build cumulative histogram
		float *pHist = hist.ptr<float>(0);
		for (size_t i = 1; i < (size_t)histSize; ++i)
		{
			pHist[i] += pHist[i - 1];
		}
		float thresh = float(percentile * pHist[histSize - 1]);
#if PERCENTILE
		std::cout << "the percentile is: " << percentile << ", the hist[size] is: " << pHist[histSize-1] << " and the threshold is: " << thresh << std::endl;
		std::cout << "running over the histogram..." << std::endl;
		for (size_t i = 1; i < (size_t)histSize; ++i)
		{
			std::cout << "hist in place " << i << " is: " << pHist[i] << std::endl;
		}
#endif
		for (size_t i = 1; i < (size_t)histSize; ++i)
		{
			if (pHist[i] >= thresh)
			{
				return (i / (float)histSize) * range[1];
			}
		}

		return 0.f; // this line only cancels a compiler warning and should never be reached.
	}
	//__________________________________________________________________________________________________
#define ROUND(x) (static_cast<int>(x + 0.5))
	void depthImageToWorldCoord_depth(const cv::Mat& img, const CameraIntrinsics& intrinsics, std::vector<DepthImageCoords>& coords,
		const cv::Point offset = cv::Point(0, 0))
	{
		cv::Mat nonZeroPoints;
		cv::findNonZero(img > ZERO_EPSILON, nonZeroPoints);
		auto numNonZeros = nonZeroPoints.total();
		cv::Point* pNnzPoints = (cv::Point*)nonZeroPoints.data;
		if (pNnzPoints == nullptr) // for KW
		{
			return;
		}
		coords.resize(numNonZeros);
		for (uint32_t i = 0; i < numNonZeros; ++i)
		{
			coords[i].u = (float)pNnzPoints[i].x;
			coords[i].v = (float)pNnzPoints[i].y;

			coords[i].z = img.at<float>(ROUND(coords[i].v), ROUND(coords[i].u));

			coords[i].x = coords[i].z *  (coords[i].u + offset.x - intrinsics.px) / intrinsics.fx;
			coords[i].y = coords[i].z *  (coords[i].v + offset.y - intrinsics.py) / intrinsics.fy;
		}
	}

	//__________________________________________________________________________________________________
	void ransacPlaneFitting(const cv::Mat& img_in, const CameraIntrinsics& intrinsics, const cv::Point& topLeftOffset, FittedPlane& res,
		const RansacParams& params, std::vector<DepthImageCoords>& nnzCoords)
	{
		cv::Mat img;
		img_in.convertTo(img, CV_32FC1);
		// Perform RANSAC on the visible pixels so we can model them as a surface
		depthImageToWorldCoord_depth(img, intrinsics, nnzCoords, topLeftOffset);

		// Set inlierThresh
		float range[] = { (float)ZERO_EPSILON, 65535.f };
		//double percentilePoint = computePercentilePoint(img, params.percentilePoint, range);
		//res.inlierThreshold = percentile2InlierThresh(percentilePoint) * params.inlierThreshFactor;

		uint32_t bestNumInlier = 0;
		std::vector<double> bestParams(3, 0.0f);
		auto numSamples = nnzCoords.size();
		cv::Mat mat(3, 3, CV_64F);
		double* pMat = (double*)mat.data;

		cv::RNG randNumGen(cv::getTickCount());

		for (uint32_t iter = 0; iter < params.numIterations; ++iter)
		{

			std::vector<double> depthVals(3);
			uint32_t numInlier = 0;
			std::vector<double> currentParams(3);

			// 1. Create 3X3 mat = [X, Y, 1]
			for (uint32_t i = 0; i < 3; ++i)
			{
				// Sample random visible pixel
				uint32_t randIndx = randNumGen.uniform(0, (int)numSamples);
				depthVals[i] = nnzCoords[randIndx].z;
				pMat[i * 3] = nnzCoords[randIndx].x;
				pMat[i * 3 + 1] = nnzCoords[randIndx].y;
				pMat[i * 3 + 2] = 1.0f;
			}

			// the inlier threshold is the average of the sampled points' z values
			double sumZ = 0, avgZ = 0;
			for (auto& val : depthVals)
				sumZ += val;
			avgZ = sumZ / 3.0;
			res.inlierThreshold = percentile2InlierThresh(avgZ) * params.inlierThreshFactor;

			// 2. Calculate the current plane params verifying { currentParams * mat = depthVals }
			cv::Mat invMat = mat.inv();
			double* pInvMat = (double*)invMat.data;
			for (uint32_t i = 0; i < 3; ++i)
			{
				double currVal = 0.0f;
				for (uint32_t j = 0; j < 3; ++j)
				{
					currVal += pInvMat[i * 3 + j] * depthVals[j];
				}
				currentParams[i] = currVal;
			}

			// 3. Count the number of inliers defined by diff = | Z_all - [X_whole, Y_whole, 1] * currentParams | < Thresh
			for (uint32_t i = 0; i < nnzCoords.size(); ++i)
			{
				double estimatedDepth = nnzCoords[i].x * currentParams[0] + nnzCoords[i].y * currentParams[1] + currentParams[2];
				double diff = std::fabs(nnzCoords[i].z - estimatedDepth);

				if (diff < res.inlierThreshold)
				{
					++numInlier; // This is an inlier
				}
			}

			if (numInlier > bestNumInlier)
			{
				// We have a better estimation
				bestNumInlier = numInlier;
				bestParams = currentParams;
			}

			if (bestNumInlier == numSamples)
			{
				break;
			}
		} // for iter

		// Normalize the normal
		double NormalNorm = std::sqrt(bestParams[0] * bestParams[0] + bestParams[1] * bestParams[1] + 1);
		res.planeEquation[0] = bestParams[0] / NormalNorm;
		res.planeEquation[1] = bestParams[1] / NormalNorm;
		res.planeEquation[2] = -1.0 / NormalNorm;
		res.planeEquation[3] = bestParams[2] / NormalNorm;
		
		// if the plane is the vertical wall (and not the shelf) extend the threshold
		if (res.planeEquation[3] >= 600)
			res.inlierThreshold *= 10;

		// 4. Store the best plane inliers
		res.inliersMask = cv::Mat::zeros(img.size(), CV_8U);

		for (uint32_t i = 0; i < nnzCoords.size(); ++i)
		{
			double estimatedDepth = nnzCoords[i].x * bestParams[0] + nnzCoords[i].y * bestParams[1] + bestParams[2];
			double diff = std::abs(nnzCoords[i].z - estimatedDepth);
			if (diff < res.inlierThreshold)
			{
				res.inliersMask.at<unsigned char>((int)nnzCoords[i].v, (int)nnzCoords[i].u) = 1;
				res.inliers.emplace_back(nnzCoords[i].worldCoords());
			}
		}

		// 5. Refine plane params -- Use SVD to refine normals on the inliers previously found
		// NOTE TO HELING: this does not appear to improve the results
		if (params.refinePlane)
		{
#ifdef OB_DEBUG2_
			std::cout << "Plane equation before refinement: ("
				<< res.planeEquation[0] << ", "
				<< res.planeEquation[1] << ", "
				<< res.planeEquation[2] << ", "
				<< res.planeEquation[3] << ")"
				<< std::endl;
#endif
			cv::Mat XYZ_zeroMean((int)res.inliers.size(), 3, CV_64F);
			double *pXYZ = (double*)XYZ_zeroMean.data;
			for (unsigned int i = 0; i < res.inliers.size(); ++i)
			{
				pXYZ[i * 3] = res.inliers[i].x;
				pXYZ[i * 3 + 1] = res.inliers[i].y;
				pXYZ[i * 3 + 2] = res.inliers[i].z;
			}
			// Remove mean value
			cv::Point3d centroid;
			centroid.x = cv::mean(XYZ_zeroMean.col(0)).val[0];
			centroid.y = cv::mean(XYZ_zeroMean.col(1)).val[0];
			centroid.z = cv::mean(XYZ_zeroMean.col(2)).val[0];
			XYZ_zeroMean.col(0) -= centroid.x;
			XYZ_zeroMean.col(1) -= centroid.y;
			XYZ_zeroMean.col(2) -= centroid.z;

			// Perform SVD
			cv::Mat s, u, vt;
			cv::SVDecomp(XYZ_zeroMean, s, u, vt);

			// Find minimal singular value
			double minVal = 1e10;
			unsigned int minIndx = 0;
			for (unsigned int column = 0; column < 3; ++column)
			{
				double val = s.at<double>(column, 0);
				if (val < minVal)
				{
					minVal = val;
					minIndx = column;
				}
			}

			double fractionalAnisotropy = s.at<double>(minIndx, 0) / (s.at<double>((minIndx + 1) % 3, 0) + s.at<double>((minIndx + 2) % 3, 0));
			res.score = res.inliers.size() / fractionalAnisotropy / img.total();

			res.planeEquation[0] = vt.at<double>(minIndx, 0);
			res.planeEquation[1] = vt.at<double>(minIndx, 1);
			res.planeEquation[2] = vt.at<double>(minIndx, 2);
			res.planeEquation[3] = -(res.planeEquation[0] * centroid.x + res.planeEquation[1] * centroid.y + res.planeEquation[2] * centroid.z);

			// Keep plane normal toward the camera
			if (res.planeEquation[2] > 0)
			{
				res.planeEquation[0] *= -1.0;
				res.planeEquation[1] *= -1.0;
				res.planeEquation[2] *= -1.0;
				res.planeEquation[3] *= -1.0;
			}
			
			// TODO:  check results with d = oldNormal.dot( Mean ) and with newNormal.dot(Mean) ?
#ifdef OB_DEBUG2_
			std::cout << "fractionalAnisotropy: " << fractionalAnisotropy << std::endl;

			std::cout << "Plane equation after refinement: ("
				<< res.planeEquation[0] << ", "
				<< res.planeEquation[1] << ", "
				<< res.planeEquation[2] << ", "
				<< res.planeEquation[3] << ")"
				<< std::endl;

			std::cout << "Min singular value: " << minVal << std::endl;

#endif
		}
	}
} //namespace EPV

//--------------------------------------------------------------------------------------

namespace TransformUtils

{
	cv::Mat axisAngleToRotMat(double x, double y, double z, double angle)
	{
		double c = std::cos(angle);
		double s = std::sin(angle);
		double t = 1.0 - c;
		double norm = std::sqrt(x * x + y * y + z * z);
		
		x /= norm;
		y /= norm;
		z /= norm;

		cv::Mat R(3, 3, CV_64F);

		R.at<double>(0, 0) = c + x * x * t;
		R.at<double>(1, 1) = c + y * y * t;
		R.at<double>(2, 2) = c + z * z * t;

		double tmp1 = x * y * t;
		double tmp2 = z * s;

		R.at<double>(1, 0) = tmp1 + tmp2;
		R.at<double>(0, 1) = tmp1 - tmp2;

		tmp1 = x * z * t;
		tmp2 = y * s;

		R.at<double>(2, 0) = tmp1 - tmp2;
		R.at<double>(0, 2) = tmp1 + tmp2;

		tmp1 = y * z * t;
		tmp2 = x * s;

		R.at<double>(2, 1) = tmp1 + tmp2;
		R.at<double>(1, 2) = tmp1 - tmp2;
		return R;
	}

	//--------------------------------------------------------------------------------------

	cv::Mat eulerXYZToRotMat(double x, double y, double z)
	{
		using std::cos;
		using std::sin;

		double dataRX[] =
		{
			1.0, 0.0, 0.0,
			0.0, cos(x), -sin(x),
			0.0, sin(x), cos(x)
		};

		double dataRY[] =
		{
			cos(y), 0.0, sin(y),
			0.0, 1.0, 0.0,
			sin(y), 0.0, cos(y)
		};

		double dataRZ[] =
		{
			cos(z), -sin(z), 0.0,
			sin(z), cos(z), 0.0,
			0.0, 0.0, 1.0
		};

		cv::Mat Rx(3, 3, CV_64F, dataRX);
		cv::Mat Ry(3, 3, CV_64F, dataRY);
		cv::Mat Rz(3, 3, CV_64F, dataRZ);
		return Rz * Ry * Rx;
	}

	//--------------------------------------------------------------------------------------

	void rotMatToEulerXYZ(cv::Mat R, double& x, double& y, double& z, bool inDegrees)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sqrt(R.at<double>(2, 1)*R.at<double>(2, 1) + R.at<double>(2, 2)*R.at<double>(2, 2)));
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));

		if (inDegrees)
		{
			x *= TransformUtils::RAD2DEG;
			y *= TransformUtils::RAD2DEG;
			z *= TransformUtils::RAD2DEG;
		}
	}

	//--------------------------------------------------------------------------------------

	cv::Mat quaternionToRotMat(const cv::Mat& quat)
	{
		assert(quat.total() == 4);
		return quaternionToRotMat(quat.data[0], quat.data[1], quat.data[2], quat.data[3]);
	}

	cv::Mat quaternionToRotMat(double x, double y, double z, double w)
	{
		// normalize if needed
		const double norm2 = (x * x + y * y + z * z + w * w);
		if (std::abs(norm2 - 1) > std::numeric_limits<double>::epsilon())
		{
			double norm = std::sqrt(norm2);
			x /= norm;
			y /= norm;
			z /= norm;
			w /= norm;
		}
		double x2 = x * x;
		double y2 = y * y;
		double z2 = z * z;
		double w2 = w * w;

		cv::Mat res(3, 3, CV_64F);
		double* pRot = (double*)res.data;
		int index = 0;
		pRot[index++] = 1 - 2 * y2 - 2 * z2;
		pRot[index++] = 2 * x*y - 2 * z*w;
		pRot[index++] = 2 * x*z + 2 * y*w;
		pRot[index++] = 2 * x*y + 2 * z*w;
		pRot[index++] = 1 - 2 * x2 - 2 * z2;
		pRot[index++] = 2 * y*z - 2 * x*w;
		pRot[index++] = 2 * x*z - 2 * y*w;
		pRot[index++] = 2 * y*z + 2 * x*w;
		pRot[index++] = 1 - 2 * x2 - 2 * y2;
		return res;
	}

	//--------------------------------------------------------------------------------------

	cv::Mat homogenize34(const cv::Mat& m)
	{
		cv::Mat res;
		double homoRowData[] = { 0.0, 0.0, 0.0, 1.0 };
		cv::Mat homoRow(1, 4, CV_64F, homoRowData);
		cv::vconcat(m, homoRow, res);
		return res;
	};

	//--------------------------------------------------------------------------------------

	// data is Nx3, R 3x3, t is 3x1

	cv::Mat doTransform(const cv::Mat& data, const cv::Mat& R, const cv::Mat& t)
	{
		return (R * data.t() + cv::repeat(t, 1, data.rows)).t();
	}

	//--------------------------------------------------------------------------------------

	// data is Nx3, R 3x3, t is 3x1

	cv::Mat doInverseTransform(const cv::Mat& data, const cv::Mat& Rt)
	{
		cv::Mat R = Rt(cv::Rect(0, 0, 3, 3));
		cv::Mat t = Rt(cv::Rect(3, 0, 1, 3));
		return doTransform(data, R.t(), -R.t() * t);
	}

	//--------------------------------------------------------------------------------------

	// data is Nx3, R 3x3, t is 3x1

	cv::Mat doInverseTransform(const cv::Mat& data, const cv::Mat& R, const cv::Mat& t)
	{
		//return (R.t() * (data - cv::repeat(t.t(), data.rows, 1)).t();
		return (data - cv::repeat(t.t(), data.rows, 1)) * R; // equivalent to the above, more efficient
	}

	//--------------------------------------------------------------------------------------

	// data is Nx3, Rt 3x4
	cv::Mat doTransform(const cv::Mat& data, const cv::Mat& Rt)
	{
		cv::Mat R = Rt(cv::Rect(0, 0, 3, 3));
		cv::Mat t = Rt(cv::Rect(3, 0, 1, 3));
		return doTransform(data, R, t);
	}



	//--------------------------------------------------------------------------------------
	// Compute body local axes, given 3 (unaligned) points
	cv::Mat computeAxes(const cv::Mat& ABC)
	{
		cv::Mat x, y, z;
		// X
		cv::normalize(ABC.row(1) - ABC.row(0), x);
		// Y
		y = ABC.row(2) - ABC.row(0);
		y -= (x.dot(y)) * x; // remove y's projection onto x axis
		cv::normalize(y, y);
		//Z
		z = x.cross(y);
		cv::Mat R;
		R.push_back(x);
		R.push_back(y);
		R.push_back(z);
		R = R.t();
		return R;
	}



	//--------------------------------------------------------------------------------------

	// computeBodyTransform from XYZ1 to XYZ2
	double computeBodyTransform(const cv::Mat& XYZ1, const cv::Mat& XYZ2, cv::Mat& R, cv::Mat& t)
	{
		// outputs [R|t] (3x4)
		//re - center, asssuming XYZ1 and 2 represent the same points!
		assert(XYZ1.rows == XYZ2.rows);
		int N = XYZ1.rows;
		// Re-center each point set
		const cv::Mat XYZ[2] = { XYZ1, XYZ2 };
		cv::Mat centroid[2], XYZ_centered[2];
		for (int i = 0; i < 2; ++i)
		{
			cv::reduce(XYZ[i], centroid[i], 0, CV_REDUCE_AVG);
			XYZ_centered[i] = XYZ[i] - cv::repeat(centroid[i], N, 1);
		}
		// compute covariance and do SVD
		cv::Mat H = XYZ_centered[0].t() * XYZ_centered[1];
		cv::SVD svd(H);
		// Compute R
		R = svd.vt.t() * svd.u.t();
		// special reflection case
		double det_R = cv::determinant(R);
		if (std::abs(det_R + 1.0) < 0.0001)
		{
			svd.vt.row(2) *= -1.f;
			R = svd.vt.t() * svd.u.t();
		}
		t = -R * centroid[0].t() + centroid[1].t();

		//=====================================================================

		// Do transform
		cv::Mat XYZ0_transformed = doTransform(XYZ[0], R, t);
		// residual translation
		cv::Mat diff = (XYZ0_transformed - XYZ[1]); // xi - x0i, ...
		cv::Mat avgResidualTransl;
		cv::reduce(diff, avgResidualTransl, 0, CV_REDUCE_SUM);
		//avgResidualTransl = avgResidualTransl.t();

		// compute RMS
		diff = diff.mul(diff); // (xi - x0i)^2, ...
		cv::Mat dist2;
		cv::reduce(diff, dist2, 1, CV_REDUCE_SUM);
		cv::Mat meanDist2;
		cv::reduce(dist2, meanDist2, 0, CV_REDUCE_AVG);
		double rms = (double)meanDist2.at<double>(0);
		rms = std::sqrt(rms);
		//printMat("avgResidualT (mm): ", avgResidualTransl * 1000, 2);
		//printMat("rms = ", meanDist2 * 1000, 3);
		return rms;
	}
	//--------------------------------------------------------------------------------------

	double computeAvgDistance(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& dist)
	{
		cv::Mat diff = (m1 - m2);
		diff = diff.mul(diff);
		cv::reduce(diff, dist, 1, CV_REDUCE_SUM);
		cv::sqrt(dist, dist);
		cv::Mat avgDist;
		cv::reduce(dist, avgDist, 0, CV_REDUCE_AVG);
		return avgDist.at<double>(0);
	}
	//--------------------------------------------------------------------------------------

	void matchPointSets(const cv::Mat XYZ1, const cv::Mat XYZ2, float maxDistance, cv::Mat& res1, cv::Mat& res2)
	{
		// Clear output mats
		res1 = cv::Mat();
		res2 = cv::Mat();
		// Match
		std::vector<std::vector<cv::DMatch> > matches;
		cv::Mat XYZ_cpy[2];
		XYZ1.convertTo(XYZ_cpy[0], CV_32F);
		XYZ2.convertTo(XYZ_cpy[1], CV_32F);
		cv::BFMatcher matcher(cv::NORM_L2, true);
		matcher.radiusMatch(XYZ_cpy[0], XYZ_cpy[1], matches, maxDistance);

		for (auto sample : matches)
		{
			if (sample.empty())
			{
				//std::cout << "empty"<< std::endl;
				continue;
			}
			else
			{
				auto match = sample[0]; //take closest
				res1.push_back(XYZ1.row(match.queryIdx));
				res2.push_back(XYZ2.row(match.trainIdx));
				/*
				std::cout << match.queryIdx << ", "
				<< match.trainIdx << ", "
				<< match.distance<< ", "
				<<std::endl;
				*/
			}
		}
	}
	//---------------------------------------

	cv::Mat projectCloud(const cv::Mat& cloud, unsigned int w, unsigned int h, double fx, double fy, double cx, double cy, bool inMM)
	{
		cv::Mat depth(h, w, CV_16UC1, cv::Scalar(0));
		double* pCloud = (double*)cloud.data;
		for (int p = 0; p < cloud.rows; ++p, pCloud += 3)
		{
			double x = pCloud[0];
			double y = pCloud[1];
			double z = pCloud[2];
			double zMM = inMM ? z : z * 1000;
			if (z < 0)
			{
				continue;
			}
			double u = fx / z * x + cx;
			double v = fy / z * y + cy;
			//std::cout << u << " " << v << std::endl;
			int uInt = (int)(u + 0.5);
			int vInt = (int)(v + 0.5);
			if (uInt >= 0 && vInt >= 0 && uInt < (int)w && vInt < (int)h)
			{
				unsigned short& depthVal = depth.at<unsigned short>(vInt, uInt);
				if (depthVal == 0 || depthVal > zMM)
				{
					depthVal = (unsigned short)(zMM + 0.5);
				}
			}
		}
		return depth;
	}
	//---------------------------------------

	// Input
	cv::Mat projectColoredCloud(const cv::Mat& cloud, const cv::Mat& rgb, unsigned int w, unsigned int h, double fx, double fy, double cx, double cy, bool inMM)
	{
		cv::Mat depth(h, w, CV_16UC1, cv::Scalar(0));
		cv::Mat img(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
		double* pCloud = (double*)cloud.data;
		double* pRgb = (double*)rgb.data;
		for (int p = 0; p < cloud.rows; ++p, pCloud += 3, pRgb += 3)
		{
			double x = pCloud[0];
			double y = pCloud[1];
			double z = pCloud[2];
			double zMM = inMM ? z : z * 1000;
			if (z < 0)
			{
				continue;
			}
			double u = fx / z * x + cx;
			double v = fy / z * y + cy;
			//std::cout << u << " " << v << std::endl;
			int uInt = (int)(u + 0.5);
			int vInt = (int)(v + 0.5);
			if (uInt >= 0 && vInt >= 0 && uInt < (int)w && vInt < (int)h)
			{
				unsigned short& depthVal = depth.at<unsigned short>(vInt, uInt);
				if (depthVal == 0 || depthVal > zMM)
				{
					depthVal = (unsigned short)(zMM + 0.5);
					for (int channel = 0; channel < 3; channel++)
					{
						img.at<cv::Vec3b>(vInt, uInt)[2 - channel] = (unsigned char)pRgb[channel];
					}
				}
			}
		}
		return img;
	}
	//---------------------------------------

	cv::Mat unprojectColoredCloud(const cv::Mat& depth, const cv::Mat& rgb, double fx, double fy, double cx, double cy, int minz, int maxz, bool inMM)
	{
		cv::Mat cloud(cv::countNonZero(depth), 6, CV_64F);
		double* pCloud = (double*)cloud.data;
		int numPoints = 0;
		unsigned short* pDepth = (unsigned short*)depth.data;
		for (int i = 0; i < depth.rows; ++i)
		{
			for (int j = 0; j < depth.cols; ++j, pDepth++)
			{
				double z = (double)pDepth[0];
				if (!inMM)
					z *= 0.001; // mm to meters
				if (z <= 0)
				{
					continue;
				}
				if (z < minz || z > maxz)
				{
					continue;
				}
				//// TODO remove
				//if (j > 594 || j < 367
				//        || (i > 328 || i < 155))
				//{
				//    continue;
				//}
				pCloud[0] = ((j - cx) * z / fx);
				pCloud[1] = ((i - cy) * z / fy);
				pCloud[2] = z;
				pCloud[3] = (double)rgb.at<cv::Vec3b>(i, j)[2];
				pCloud[4] = (double)rgb.at<cv::Vec3b>(i, j)[1];
				pCloud[5] = (double)rgb.at<cv::Vec3b>(i, j)[0];
				pCloud += 6;
				numPoints++;
			}
			cloud.resize(numPoints);
		}
		return cloud;
	}

	//---------------------------------------

	cv::Mat unprojectCloud(const cv::Mat& depth, double fx, double fy, double cx, double cy, int minz, int maxz, bool inMM)
	{

		cv::Mat cloud(cv::countNonZero(depth), 3, CV_64F);
		double* pCloud = (double*)cloud.data;
		int numPoints = 0;
		unsigned short* pDepth = (unsigned short*)depth.data;
		for (int i = 0; i < depth.rows; ++i)
		{
			for (int j = 0; j < depth.cols; ++j, pDepth++)
			{
				double z = (double)pDepth[0];
				if (z <= 0)
				{
					continue;
				}
				if (z < minz || z > maxz)
				{
					continue;
				}
				if(!inMM)
					z *= 0.001; // mm to meters
				pCloud[0] = ((j - cx) * z / fx);
				pCloud[1] = ((i - cy) * z / fy);
				pCloud[2] = z;
				pCloud += 3;
				numPoints++;
			}
			cloud.resize(numPoints);
		}
		return cloud;
	}

	//---------------------------------------

	cv::Mat createTriadCloud()
	{
		cv::Mat c;
		int numPoints = 100;
		double unitLength = 1.0;
		c.push_back(cv::Mat(1, 3, CV_64F, cv::Scalar(0)));
		for (int coord = 0; coord < 3; coord++, unitLength += 1.0)
		{
			cv::Mat p(1, 3, CV_64F, cv::Scalar(0));
			double dL = unitLength / numPoints;
			for (int i = 0; i < numPoints; i++)
			{
				p.at<double>(0, coord) += dL;
				c.push_back(p);
			}
		}
		return c;
	}
}

//--------------------------------------------------------------------------------------

