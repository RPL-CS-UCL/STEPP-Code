#!/usr/bin/python3

import cv2
import numpy as np
from camera import Camera
from scipy.spatial.transform import Rotation as R
import rospy
from nav_msgs.msg import Odometry as odom
import os
import matplotlib.pyplot as plt
import json


class CameraPinhole(Camera):
    def __init__(self, width, height, camera_name, distortion_model, K, D, Rect, P):
        super().__init__(width, height, camera_name, distortion_model, K, D, Rect, P)

    def undistort(self, image):
          undistorted_image = cv2.undistort(image, self.K, self.D)
          return undistorted_image

def main():
    """Main function to test the Camera class."""
    # Create a pinhole camera model
    D = np.array([-0.28685832023620605, -2.0772109031677246, 0.0005875344504602253, -0.0005043392884545028, 1.5214914083480835, -0.39617425203323364, -1.8762085437774658, 1.4227665662765503])
    K = np.array([607.9638061523438, 0.0, 638.83984375, 0.0, 607.9390869140625, 367.0916748046875, 0.0, 0.0, 1.0]).reshape(3, 3)
    Rect = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    P = np.array([607.9638061523438, 0.0, 638.83984375, 0.0, 0.0, 607.9390869140625, 367.0916748046875, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)
    camera_pinhole = CameraPinhole(width=1280, height=720, camera_name='kinect_camera', 
                                distortion_model='rational_polynomial', 
                                K=K, D=D, Rect=Rect, P=P)

    # Initialize lists to store coordinates and orientations
    coordinates = []
    orientations = []
    directions = []

    folder_path = '/home/sebastian/Documents/ANYmal_data/OPS_grass/odom_chosen_images_2/'
    images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))])
    img_file_names = [os.path.basename(img) for img in images]

    # Initialize the ROS node
    rospy.init_node('trajectory_publisher', anonymous=True)
    # Initialize the publisher
    pub = rospy.Publisher('/trajectory', odom, queue_size=10)
    pub2 = rospy.Publisher('/trajectory2', odom, queue_size=10)

    # Load the coordinates and orientations
    coordinates_path = '/home/sebastian/Documents/code/Trajectory_extract/odom_data.txt'
    # coordinates_path = '/home/sebastian/Documents/code/Trajectory_extract/odom_data_halfsecond.txt'

    T_odom_list = []
    with open(coordinates_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comment lines
            parts = line.split()
            if parts:
                coordinates.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
                # print(coordinates[-1])
                orientations.append(np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])) #qx, qy, qz, qw
                # print(orientations[-1])

                T_odom = np.eye(4, 4)
                T_odom[:3, :3] = R.from_quat(orientations[-1]).as_matrix()[:3, :3]
                T_odom[:3, 3] = coordinates[-1]
                T_odom_list.append(T_odom)

    #difference between odometry frame and camera frame
    translation = [-0.739, -0.056, -0.205] #x, y, z
    path_translation = [0.0, 1.0, 0.0] #x, y, z
    rotation = [0.466, -0.469, -0.533, 0.528] #quaternion
    T_imu_camera = np.eye(4, 4)
    T_imu_camera[:3, :3] = R.from_quat(rotation).as_matrix()[:3, :3]
    T_imu_camera[:3, 3] = translation

    # rotation = [-0.469, -0.533, 0.528, 0.466] #quaternion

    for i in range(len(coordinates)):
        T_world_camera = np.linalg.inv(T_imu_camera) @ T_odom_list[i] @ T_imu_camera

        coordinates[i] = T_world_camera[:3, 3]
        orientations[i] = R.from_matrix(T_world_camera[:3, :3]).as_quat()

    #create a list of odometry messages from the coord and orientation lists
    for i in range(len(coordinates)):
        # Create a new odometry message
        odom_msg = odom()
        # Set the header
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"

        # Set the position
        odom_msg.pose.pose.position.x = coordinates[i][0]
        odom_msg.pose.pose.position.y = coordinates[i][1]
        odom_msg.pose.pose.position.z = coordinates[i][2]
        # Set the orientation
        odom_msg.pose.pose.orientation.x = orientations[i][0]
        odom_msg.pose.pose.orientation.y = orientations[i][1]
        odom_msg.pose.pose.orientation.z = orientations[i][2]
        odom_msg.pose.pose.orientation.w = orientations[i][3]
        # Append the message to the list
        directions.append(odom_msg)

    # publish data to ros topic
    # for i in range(len(coordinates)):
    #     # Publish the message
    #     pub.publish(directions[i])
    #     # Sleep for 0.1 seconds
    #     rospy.sleep(0.01)
    #     print(f"Published message {i+1}/{len(coordinates)}", end='\r')

        # if i == point:
        #     pub2.publish(directions[i])

    def unit_vector(vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return vector
        return vector / magnitude

    def trasnform_coord(quat, coord):
        R1 = R.from_quat(quat).as_matrix()
        #transpose R1 to get the inverse
        return R1.T @ coord

    def translate_to_frame(coords, point, quat):
        # for a given coordinate and orientation pair
        New_frame_coord = []
        for i in range(1, len(coords)):
            c = trasnform_coord(quat, point - coords[i] - path_translation)
            New_frame_coord.append(c)
        # print('\n c:',c)
        return np.array(New_frame_coord)  
    
    # #create cv2 window
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1280, 720)

    u_C2_past = np.zeros((2, 1))
    save_flag = True
    img_points = []

    future_steps = 50
    all_points = []

    # exit()
    print('length of coordinates:', len(coordinates))

    for i in range(1, len(coordinates)- future_steps):
        point = i
        points = translate_to_frame(coordinates[point:], coordinates[point], orientations[point])
        p_C2 = points.T
        # Project a 3D point into the pixel plane
        #make the point number a 6 digit string
        img = cv2.imread(folder_path + img_file_names[point])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        img_points = []

        u_C2_past[0] = 1280/2
        u_C2_past[1] = 720

        for j in range(1, future_steps):
            p_C = p_C2[:, j]
            _, tmp_p = camera_pinhole.project(p_C)
            tmp_p = tmp_p.reshape(2, 1)
            u_C1 = tmp_p[:, 0]
            tmp_p, _ = cv2.projectPoints(p_C.reshape(1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, D)
            u_C2 = tmp_p[0, 0, :2]
            if u_C2[0] < camera_pinhole.width and u_C2[0] > 30 and u_C2[1] < camera_pinhole.height-20 and u_C2[1] > 0:

                # set points to be drawn on the image
                cv2.circle(img, (int(u_C2[0]), int(u_C2[1])), 5, (0, 0, 255), -1) 
                cv2.line(img, (int(u_C2_past[0]), int(u_C2_past[1])), (int(u_C2[0]), int(u_C2[1])), (255 - j*(255/future_steps), j*(255/future_steps), 0), 2)    # green line

                #append img_points to img_points
                img_points.append([int(u_C2[0]),int(u_C2[1])])

            u_C2_past = u_C2
        # print(img_points)
        #append img_points to all_points as another dimension

        all_points.append(img_points)

        # Display the image with the points drawn on it in the cv2 window
        cv2.imshow('image', img)
        cv2.waitKey(0)

        print(f"Point {point}/{len(coordinates)}", end='\r')
    
    #save all_points to a numpy file
    # print(all_points[-7:])
    print(len(all_points))
    
    #save all_pints as a json
    with open('OPS_grass_pixels.json', 'w') as f:
        json.dump(all_points, f)


if __name__ == '__main__':
    main()