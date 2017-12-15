#include "ros/ros.h"
#include "tf/tf.h"


#include "refills_msgs/CountObjects.h"
#include "refills_counting/RealSenseBridge.h"
#include "volume_counter/counter.h"

class CounterInterface
{
  ros::NodeHandle nh_;
  RealSenseBridge cameraBridge_;
  ros::ServiceServer service;

  counter productCounter;

  bool first;
public:
  CounterInterface(ros::NodeHandle n): nh_(n), cameraBridge_(nh_), first(true)
  {
    service = nh_.advertiseService("count", &CounterInterface::cb_, this);
  }

  bool cb_(refills_msgs::CountObjects::Request  &req,
           refills_msgs::CountObjects::Response &res)
  {
    ROS_INFO("request: depth=%ld, height=%ld", (long int)req.obj_depth, (long int)req.obj_height);
    ROS_INFO("Waiting for camera data");
    while(!cameraBridge_.newData())
    {
      usleep(100);
    }
    cv::Mat rgb, depth;
    sensor_msgs::CameraInfo cam_info;
    cameraBridge_.getData(rgb, depth, cam_info);
    if(first)
    {
      //set camera params in productCounter
      first=false;
    }

    tf::Stamped<tf::Pose> poseStamped;
    tf::poseStampedMsgToTF(req.separator_location,poseStamped);
    tf::Matrix3x3 rotMat = poseStamped.getBasis();
    cv::Mat cvRot(3, 3, CV_64F);
    cvRot.at<double>(0,0) = rotMat[0][0]; cvRot.at<double>(0,1) = rotMat[0][1]; cvRot.at<double>(0,2) = rotMat[0][2];
    cvRot.at<double>(1,0) = rotMat[1][1]; cvRot.at<double>(1,1) = rotMat[1][1]; cvRot.at<double>(1,2) = rotMat[1][2];
    cvRot.at<double>(2,0) = rotMat[2][1]; cvRot.at<double>(2,1) = rotMat[2][1]; cvRot.at<double>(2,2) = rotMat[2][2];
    productCounter.setRotation(cvRot);
    productCounter.setImg(depth);
    cv::Point3d separatorPosition;
    separatorPosition.x = poseStamped.getOrigin().x();
    separatorPosition.y = poseStamped.getOrigin().y();
    separatorPosition.z = poseStamped.getOrigin().z();
    productCounter.setSeparator(separatorPosition);
    productCounter.setType(req.obj_width,req.obj_height,req.obj_depth,req.object_name);

    res.object_count = productCounter.countObjects();
    ROS_INFO("sending back response: [%ld]", (long int)res.object_count);
    return true;
  }
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "count_objects_node");
  ros::NodeHandle nh("~");

  CounterInterface ci(nh);

  ros::spin();
  return 0;
}
