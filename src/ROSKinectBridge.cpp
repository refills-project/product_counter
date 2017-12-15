/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author(s): Ferenc Balint-Benczedi <balintbe@cs.uni-bremen.de>
 *         Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *         Jan-Hendrik Worch <jworch@cs.uni-bremen.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <refills_counting/ROSKinectBridge.h>

//OpenCV
#include <cv_bridge/cv_bridge.h>

ROSKinectBridge::ROSKinectBridge() : nodeHandle("~"),it(nodeHandle),spinner(0),_newData(false)
{
  config();
  initSpinner();
}
ROSKinectBridge::ROSKinectBridge(ros::NodeHandle nh) : nodeHandle(nh),it(nodeHandle),spinner(0),_newData(false)
{
  config();
  initSpinner();
}

ROSKinectBridge::~ROSKinectBridge()
{
  spinner.stop();
  delete sync;
  delete rgbImageSubscriber;
  delete depthImageSubscriber;
  delete cameraInfoSubscriber;
}

void ROSKinectBridge::initSpinner()
{
  sync = new message_filters::Synchronizer<RGBDSyncPolicy>(RGBDSyncPolicy(5), *rgbImageSubscriber, *depthImageSubscriber, *cameraInfoSubscriber);
  sync->registerCallback(boost::bind(&ROSKinectBridge::cb_, this, _1, _2, _3));
  spinner.start();
}

void ROSKinectBridge::config()
{
  std::string depth_topic = "/camera/depth/image_raw";
  std::string color_topic = "/camera/color/image_raw";
  std::string depth_hints = "compressedDepth";
  std::string color_hints = "compressed";
  std::string cam_info_topic = "/camera/color/camera_info";

  depthOffset = 0;
  scale=false;

  image_transport::TransportHints hintsColor(color_hints);
  image_transport::TransportHints hintsDepth(depth_hints);

  depthImageSubscriber = new image_transport::SubscriberFilter(it, depth_topic, 1, hintsDepth);
  rgbImageSubscriber = new image_transport::SubscriberFilter(it, color_topic, 1, hintsColor);
  cameraInfoSubscriber = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nodeHandle, cam_info_topic, 1);

  ROS_INFO("  Depth topic: [%s]", depth_topic.c_str());
  ROS_INFO("  Color topic: [%s]", color_topic.c_str());
  ROS_INFO("  CamInfo topic: [%s]", cam_info_topic.c_str());
  ROS_INFO("  Depth Hints: [%s]", depth_hints.c_str());
  ROS_INFO("  Color Hints: [%s]", color_hints.c_str());
}

void ROSKinectBridge::cb_(const sensor_msgs::Image::ConstPtr rgb_img_msg,
                          const sensor_msgs::Image::ConstPtr depth_img_msg,
                          const sensor_msgs::CameraInfo::ConstPtr camera_info_msg)
{

  cv::Mat color, depth;
  sensor_msgs::CameraInfo cameraInfo, cameraInfoHD;

  cv_bridge::CvImageConstPtr orig_rgb_img;
  orig_rgb_img = cv_bridge::toCvShare(rgb_img_msg, sensor_msgs::image_encodings::BGR8);
  cameraInfo = sensor_msgs::CameraInfo(*camera_info_msg);

  cv_bridge::CvImageConstPtr orig_depth_img;
  orig_depth_img = cv_bridge::toCvShare(depth_img_msg, depth_img_msg->encoding);

  if(depth_img_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1)
  {
    depth = orig_depth_img->image.clone();
  }
  else if(depth_img_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    orig_depth_img->image.convertTo(depth, CV_16U, 0.001);
  }
  else
  {
    ROS_ERROR("Unknown depth image type!");
    return;
  }

  color = orig_rgb_img->image.clone();
  lock.lock();

  this->color = color;
  this->depth = depth;
  this->cameraInfo = cameraInfo;
  _newData = true;

  lock.unlock();
}

bool ROSKinectBridge::getData(cv::Mat &rgb,cv::Mat &d, sensor_msgs::CameraInfo &cameraInfo)
{
  if(!newData())
  {
    return false;
  }

  lock.lock();
  rgb = this->color;
  d = this->depth;
  cameraInfo = this->cameraInfo;
  lock.unlock();
  return true;
}
