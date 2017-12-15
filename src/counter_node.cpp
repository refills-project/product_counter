#include "ros/ros.h"
#include "refills_msgs/CountObjects.h"
#include "refills_counting/RealSenseBridge.h"


class CounterInterface
{
  ros::NodeHandle nh_;
  RealSenseBridge cameraBridge_;
  ros::ServiceServer service;

public:
  CounterInterface(ros::NodeHandle n): nh_(n), cameraBridge_(nh_)
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
    ROS_INFO("Got Image: width [%d] height[%d]",rgb.rows,rgb.cols);
    //call algorithm etc.
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
