#include <ros/ros.h>
#include <string>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <pcl/common/eigen.h>
#include <tf/transform_datatypes.h>
#include <pcl/filters/voxel_grid.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <seb_trav_ros/Float32Stamped.h>

using namespace std;

const double PI = 3.1415926;
double depthCloudTime = 0.0;
double systemInitTime = 0;
bool systemInited = false;
bool firstLoss = true;
bool firstStampedLoss = true;
bool newDepthCloud = false;
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;
float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;
float sinVehicleRoll = 0, cosVehicleRoll = 0;
float sinVehiclePitch = 0, cosVehiclePitch = 0;
float sinVehicleYaw = 0, cosVehicleYaw = 0;
<<<<<<< HEAD
float voxel_size_ = 0.03;
double noDecayDis = 8.0;
double minDis = 2.0;
double clearingDis = 5.0;
double vehicleHeight = 1.0;
=======
float voxel_size_ = 0.1;
double noDecayDis = 5.0;
double minDis = 1.5;
double clearingDis = 3.0;
double vehicleHeight = 0.5;
>>>>>>> 1fece960f8db8bd08782cc082ed9f25b23ae8ea4
double decayTime = 8.0;
double height = 720;
double width = 1280;
float fovy;
float fovx;
<<<<<<< HEAD
float azimuth_buff = 50;
=======
float azimuth_buff = 0.0;
>>>>>>> 1fece960f8db8bd08782cc082ed9f25b23ae8ea4
int rows = 1, cols = 1;
int row_stride = 1, col_stride = 1;

Eigen::Matrix4f cameraToMapTransform;
ros::Publisher cloudPub;

pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilter;

struct CameraIntrinsics {
    double fx;
    double fy;
    double cx;
    double cy;
};

CameraIntrinsics intrinsics;
tf::Transform odomTransform;
std_msgs::Float32MultiArray loss;
seb_trav_ros::Float32Stamped losStamped;

pcl::PointCloud<pcl::PointXYZINormal>::Ptr 
    cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr 
    sparseCloud(new pcl::PointCloud<pcl::PointXYZINormal>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr 
    transformedCloud(new pcl::PointCloud<pcl::PointXYZINormal>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr 
    terrainCloud(new pcl::PointCloud<pcl::PointXYZINormal>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr
    sparseTerrainCloud(new pcl::PointCloud<pcl::PointXYZINormal>);
pcl::PointCloud<pcl::PointXYZINormal>::Ptr
    currentCloud(new pcl::PointCloud<pcl::PointXYZINormal>);
pcl::PointCloud<pcl::PointXYZI>::Ptr
    pubCloud(new pcl::PointCloud<pcl::PointXYZI>);

void setCameraIntrinsics(const std::string& cameraType) {
    ROS_INFO("Setting camera intrinsics for %s camera", cameraType.c_str());
    if (cameraType == "D455") {
        intrinsics = {634.3491821289062, 632.8595581054688, 631.8179931640625, 375.0325622558594};
        height = 720;
        width = 1280;
    } else if (cameraType == "zed2") {
        intrinsics = {534.3699951171875, 534.47998046875, 477.2049865722656, 262.4590148925781};
        height = 540;
        width = 960-2*azimuth_buff;
    } else if (cameraType == "cmu_sim") {
        intrinsics = {205.46963709898583, 205.46963709898583, 320.5, 180.5};
        height = 360;
        width = 640-2*azimuth_buff;
    } else {
        ROS_ERROR("Invalid camera type specified. Please choose from 'D455', 'zed2', or 'cmu_sim'.");
        ros::shutdown();
    }

    fovy = 2 * atan(height / (2 * intrinsics.fy));
    fovx = 2 * atan(width / (2 * intrinsics.fx));
}

// Convert 2D pixel coordinates to 3D point
pcl::PointXYZ convertTo3DPoint(int u, int v, float depth, const CameraIntrinsics& intrinsics) {
    pcl::PointXYZ point;
    point.z = depth;
    point.x = (u - intrinsics.cx) / intrinsics.fx * depth;
    point.y = (v - intrinsics.cy) / intrinsics.fy * depth;
    return point;
}

void callback(const sensor_msgs::Image::ConstPtr& depthMsg,
              const nav_msgs::Odometry::ConstPtr&  odomMsg,
              const seb_trav_ros::Float32StampedConstPtr& customMsg) {
    
    // if (loss.data.empty()) {  // Check if the loss data is not initialized
    //     ROS_WARN("Loss data not available yet.");
    //     return;  // Skip this callback cycle
    // }
    if (firstStampedLoss) {
        rows = customMsg->data.layout.dim[0].size;
        cols = customMsg->data.layout.dim[1].size;
        row_stride = customMsg->data.layout.dim[0].stride;
        col_stride = customMsg->data.layout.dim[1].stride;
        firstStampedLoss = false;
    }
    // losStamped = *customMsg;

    // Extract the position and orientation from the odometry message
    double roll, pitch, yaw;
    geometry_msgs::Point position = odomMsg->pose.pose.position;
    geometry_msgs::Quaternion orientation = odomMsg->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(orientation.x, orientation.y, orientation.z, orientation.w))
      .getRPY(roll, pitch, yaw);

    vehicleX = odomMsg->pose.pose.position.x;
    vehicleY = odomMsg->pose.pose.position.y;
    vehicleZ = odomMsg->pose.pose.position.z;

    //temp [7.251, -10.919, -3.618]
    // vehicleX = vehicleX - 7.251;
    // vehicleY = vehicleY + 10.919;
    // vehicleZ = vehicleZ + 3.618;

    vehicleRoll = roll;
    vehiclePitch = pitch;
    vehicleYaw = yaw;

    sinVehicleRoll = sin(vehicleRoll);
    cosVehicleRoll = cos(vehicleRoll);
    sinVehiclePitch = sin(vehiclePitch);
    cosVehiclePitch = cos(vehiclePitch);
    sinVehicleYaw = sin(vehicleYaw);
    cosVehicleYaw = cos(vehicleYaw);

    // Convert the position and orientation into a transform
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(position.x, position.y, position.z));
    tf::Quaternion quat(orientation.x, orientation.y, orientation.z, orientation.w);
    transform.setRotation(quat);

    // Store the transformation to be used when processing the point cloud
    odomTransform = transform;

    // Extract the depth image from the depth message
    depthCloudTime = depthMsg->header.stamp.toSec();

    if (!systemInited) {
        systemInitTime = depthCloudTime;
        systemInited = true;
    }

    cloud->clear();
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(depthMsg, depthMsg->encoding);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (depthMsg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        for (int v = 0; v < depthMsg->height; ++v) {
            for (int u = azimuth_buff; u < depthMsg->width-azimuth_buff; ++u) {
                float depth = cv_ptr->image.at<float>(v, u); // Access the depth value as float (meters)
                if (depth > 0) {  // Check for valid depth
                    pcl::PointXYZ point = convertTo3DPoint(u, v, depth, intrinsics);
                    pcl::PointXYZINormal iPoint;
                    iPoint.x = point.x;
                    iPoint.y = point.y;
                    iPoint.z = point.z;
                    iPoint.intensity = systemInitTime - depthCloudTime;;
                    iPoint.curvature = customMsg->data.data[v * row_stride + u * col_stride];
                    cloud->points.push_back(iPoint);
                }
            }
        }
    } else if (depthMsg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        for (int v = 0; v < depthMsg->height; ++v) {
            for (int u = azimuth_buff; u < depthMsg->width-azimuth_buff; ++u) {
                uint16_t depth_mm = cv_ptr->image.at<uint16_t>(v, u); // Access the depth value as uint16_t
                float depth = depth_mm * 0.001f; // Convert millimeters to meters
                if (depth != 0) {  // Check for valid depth
                    pcl::PointXYZ point = convertTo3DPoint(u, v, depth, intrinsics);
                    pcl::PointXYZINormal iPoint;
                    iPoint.x = point.x;
                    iPoint.y = point.y;
                    iPoint.z = point.z;
                    iPoint.intensity = depthCloudTime - systemInitTime;
                    iPoint.curvature = customMsg->data.data[v * row_stride + u * col_stride];
                    cloud->points.push_back(iPoint);
                }
            }
        }
    } else {
        ROS_ERROR("Unsupported depth encoding: %s", depthMsg->encoding.c_str());
        return;
    }
    newDepthCloud = true;
    // ROS_INFO("Input cloud size %zu", cloud->points.size());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_projection");
    ros::NodeHandle nh;

    std::string cameraType;
    nh.getParam("/depth_projection/camera_type", cameraType);
    nh.getParam("/depth_projection/decayTime", decayTime);
    setCameraIntrinsics(cameraType);

    // Set up subscribers using message_filters
    message_filters::Subscriber<sensor_msgs::Image> depthSub(nh, "/camera/aligned_depth_to_color/image_raw", 1);
    message_filters::Subscriber<nav_msgs::Odometry> odomSub(nh, "/state_estimation", 1);
    message_filters::Subscriber<seb_trav_ros::Float32Stamped> customMsgSub(nh, "/inference/results_stamped_post", 1);

    // Create ApproximateTime policy
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry, seb_trav_ros::Float32Stamped> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depthSub, odomSub, customMsgSub);
    sync.setInterMessageLowerBound(ros::Duration(1.5)); // Adjust time tolerance
    sync.registerCallback(boost::bind(&callback, _1, _2, _3));

    // ros::Subscriber lossSub = nh.subscribe("/inference/results", 10, lossCallback);

    // cameraToMapTransform <<   0.0, 0.0, 1.0, 0.0, // CMU_SIM transform
    //                          -1.0, 0.0, 0.0, 0.0,
    //                           0.0,-1.0, 0.0, 0.0,
    //                           0.0, 0.0, 0.0, 1.0;

    cameraToMapTransform <<  0.01165962, -0.02415892,  0.99964014,  0.482,
                            -0.99953617,  0.02784553,  0.01233136,  0.04,
                            -0.02813342, -0.99932026, -0.02382304,  0.249,
                             0.0,         0.0,         0.0,         1.0;
    
<<<<<<< HEAD
    cloudPub = nh.advertise<sensor_msgs::PointCloud2>("/depth_projection_post", 10);
=======
    cloudPub = nh.advertise<sensor_msgs::PointCloud2>("/depth_projection", 10);
>>>>>>> 1fece960f8db8bd08782cc082ed9f25b23ae8ea4

    downSizeFilter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);

    //print out the camera intrinsics
    ROS_INFO("Camera intrinsics: fx = %f, fy = %f, cx = %f, cy = %f", intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);

    ros::Rate rate(200);
    bool status = ros::ok();
    while (status) {
        ros::spinOnce();
        if (newDepthCloud) {
            newDepthCloud = false;

            //clear point clouds
            terrainCloud->clear();
            transformedCloud->clear();
            sparseCloud->clear();
            sparseTerrainCloud->clear();

            // Update terrain cloud as to get rid of old points outside decay distance
            int currentCloudSize = currentCloud->points.size();
            for (int i = 0; i < currentCloudSize; i++) {
                pcl::PointXYZINormal point = currentCloud->points[i];

                // Translate point to vehicle coordinate frame
                float translatedX = point.x - vehicleX;
                float translatedY = point.y - vehicleY;
                float translatedZ = point.z - vehicleZ;

                // Rotate point according to vehicle orientation
                float rotatedX = cosVehicleYaw * translatedX + sinVehicleYaw * translatedY;
                float rotatedY = -sinVehicleYaw * translatedX + cosVehicleYaw * translatedY;
                float rotatedZ = cosVehiclePitch * translatedZ - sinVehiclePitch * rotatedX;

                // Calculate planar distance in XY plane
                float dis = sqrt(rotatedX * rotatedX + rotatedY * rotatedY);

                // Calculate azimuth and elevation angles
                float angle1 = atan2(rotatedY, rotatedX);  // Azimuth angle
                float angle2 = atan2(rotatedZ, dis);       // Elevation angle

                // Check if the point is outside the decay time OR within no-decay distance
                // Also, check if the point is outside the FOV in both azimuth and elevation
                if ((depthCloudTime - systemInitTime + point.intensity < decayTime || dis < clearingDis)
                    && point.z < vehicleHeight
                    && (((fabs(angle1) > (fovx / 2) - 8*(PI/180) || fabs(angle2) > (fovy / 2))) || dis < minDis)) {  // Use OR instead of AND
                    terrainCloud->push_back(point);
                }
<<<<<<< HEAD
            // ROS_INFO("vehicleZ %f, vehicleHeight %f, point.z %f", vehicleZ, vehicleHeight, point.z);
=======
>>>>>>> 1fece960f8db8bd08782cc082ed9f25b23ae8ea4
            // ROS_INFO("sysinit %f, depth %f, intensity %f, time diff %f",systemInitTime, depthCloudTime, point.intensity, depthCloudTime - systemInitTime - point.intensity);
            }

            //filter the terrain cloud
            downSizeFilter.setInputCloud(terrainCloud);
            downSizeFilter.filter(*sparseTerrainCloud);

            //filter input depth cloud
            // downSizeFilter.setInputCloud(cloud);
            // downSizeFilter.filter(*sparseCloud);

            // Transform the point cloud to the map frame
            pcl::transformPointCloud(*cloud, *transformedCloud, cameraToMapTransform);

            // ROS_INFO("transformedCloud size %zu", transformedCloud->points.size());

            // Transform each point in the cloud to be in the odometry frame
            int transformedCloudSize = transformedCloud->points.size();
            for (int i =0; i < transformedCloudSize; i++) {
                pcl::PointXYZINormal point = transformedCloud->points[i];
                tf::Vector3 p(point.x, point.y, point.z);
                tf::Vector3 pTransformed = odomTransform * p;
                pcl::PointXYZINormal newPoint;
                newPoint.x = pTransformed.x();
                newPoint.y = pTransformed.y();      
                newPoint.z = pTransformed.z();
                newPoint.intensity = point.intensity;
                newPoint.curvature = point.curvature;
                float dis = sqrt((newPoint.x - vehicleX) * (newPoint.x - vehicleX) + (newPoint.y - vehicleY) * (newPoint.y - vehicleY));
                if (newPoint.z <  vehicleZ + vehicleHeight && dis > minDis && dis < noDecayDis) {
                    sparseTerrainCloud->push_back(newPoint);
                }
            }

            currentCloud = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>(*sparseTerrainCloud));

            //loop through the terrain cloud
            pubCloud->clear();
            int terrainCloudSize = sparseTerrainCloud->points.size();
            for (int i = 0; i < terrainCloudSize; i++) {
                pcl::PointXYZINormal point = sparseTerrainCloud->points[i];
                pcl::PointXYZI newPoint;
                newPoint.x = point.x;
                newPoint.y = point.y;
                newPoint.z = point.z;
                newPoint.intensity = point.curvature;
                // newPoint.intensity = point.z;
                pubCloud->push_back(newPoint);
            }

            // Publish the terrain cloud
            sensor_msgs::PointCloud2 terrainCloud2;
            pcl::toROSMsg(*pubCloud, terrainCloud2);
            terrainCloud2.header.frame_id = "odom";
            terrainCloud2.header.stamp = ros::Time().fromSec(depthCloudTime);
            cloudPub.publish(terrainCloud2);
        }

        status = ros::ok();
        rate.sleep();
    }
    
    return 0;
}