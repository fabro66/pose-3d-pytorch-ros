#!/usr/bin/env python

from __future__ import print_function
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


def image_talker(image):
    pub = rospy.Publisher("/camera/raw_image", Image, queue_size=1)
    # rospy.init_node("image_talker", anonymous=True)
    bridge = CvBridge()

    if not rospy.is_shutdown():
        pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))
