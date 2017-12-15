#include "counterUtils.h"
#include "counter.h"

double counter::countObjects()
        {
            bool splane = false, wplane = false;
            double *p, len, shelfLength = 700, cnt = 0;
            std::vector<cv::Point3d> normals(3); //  normals[1] = shelf normal (y), normals[2] = wall's (z), normals[0] = their cross product (x). 
            cv::Mat trans, nnz, srcInBox;
            std::vector<EPV::DepthImageCoords> nnzCoords; // real world coordinates

            if (_src.empty())
            {
                std::cout << "original grayscale image wasn`t loaded" << std::endl;
                return -1;
            }

            EPV::CameraIntrinsics intrinsics = { 671.062439,671.062439,679.713806,369.511169,_src.cols,_src.rows };      
            
            // 3X1 matrix for translating the coordinate system so the separator is in (0,0,0)
            cv::Mat translation(_separator);

            // NX3 matrix (point cloud) with real world coordinates for the src pixels
            cv::Rect roi = { 0,0,_src.cols,_src.rows };
            cv::Mat imgClone = _src(roi).clone();
            EPV::depthImageToWorldCoord_depth(imgClone, intrinsics, nnzCoords, roi.tl());
            nnz = cv::Mat::zeros(nnzCoords.size(), 3, CV_64FC1);
            double* nnzData = (double *)nnz.data;
            for (size_t i = 0; i < nnzCoords.size(); i++, nnzData+=3) {
                nnzData[0] = nnzCoords[i].x;
                nnzData[1] = nnzCoords[i].y;
                nnzData[2] = nnzCoords[i].z;
            }
            
            // real world coordinates after rotation and translation
            trans = TransformUtils::doInverseTransform(nnz, _rotation, translation); 
            
            // is in box
            srcInBox = cv::Mat::zeros(_src.rows, _src.cols, CV_64FC1);
            objectType object = getObject();
            for (size_t i = 0; i < trans.rows; i++)
            {
                p = trans.ptr<double>(i);
                if (p[0] >= 0 && p[0] <= object._width && p[1] <= 0 && p[1] >= object._height && p[2] >= 0 && p[2] <= shelfLength)
                    srcInBox.at<double>(nnzCoords[i].v, nnzCoords[i].u) = p[2];
            }
            cv::Mat srcInBoxF;
            srcInBox.convertTo(srcInBoxF, CV_32FC1);
            float range[] = { 1, 1025 };
            double per95 = EPV::computePercentilePoint(srcInBoxF, 0.999, range);
            double per5 = EPV::computePercentilePoint(srcInBoxF, 0.05, range);
            len = per95 - per5;
            cnt += len / object._depth;
            
            return cnt;
        }