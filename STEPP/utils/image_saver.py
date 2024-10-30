#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

class ImageSaver:
    def __init__(self, image_topic, save_directory):
        # Initialize the ROS node
        rospy.init_node('image_saver', anonymous=True)
        print('Node initialized')
        
        # Create a CvBridge object
        self.bridge = CvBridge()
        
        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber(image_topic, CompressedImage, self.image_callback)
        
        # Directory to save images
        self.save_directory = save_directory
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        
        # Counter for naming images
        self.image_counter = 0

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to a format OpenCV can work with
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
            # Create a filename for each image
            filename = os.path.join(self.save_directory, "image_{:06d}.png".format(self.image_counter))
            
            # Save the image to the specified directory
            cv2.imwrite(filename, cv_image)
            rospy.loginfo("Saved image: {}".format(filename))
            
            # Increment the counter
            self.image_counter += 1

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

if __name__ == '__main__':
    try:
        # Parameters
        image_topic = "/rgb/image_rect_color/compressed"  # Set the image topic
        save_directory = "path_to_save_folder"  # Set the directory to save images

        # Create the ImageSaver object
        image_saver = ImageSaver(image_topic, save_directory)
        
        # Keep the node running
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
