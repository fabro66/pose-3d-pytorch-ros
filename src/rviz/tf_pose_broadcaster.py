#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import rospy
import math
import tf
import numpy as np
import time
import cv2

mpii_edges = [[0, 1], [1, 2], [2, 6], [3, 6], [4, 3], [5, 4],
              [10, 11], [11, 12], [12, 8], [13, 8], [14, 13], [15, 14],
              [8, 6], [9, 8]]

joints = ["r-ankle", "r-knee", "r-hip", "l-hip", "l-knee", "l-ankle", "pelvis", "thorax", "upper neck", "head top",
          "r-wrist", "r-elbow", "r-shoulder", "l-shoulder", "l-elbow", "l-wrist"]

def show_2d_pose(img, points, c, edges=mpii_edges):
  num_joints = points.shape[0]
  points = ((points.reshape(num_joints, -1))).astype(np.int32)
  for j in range(num_joints):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  for e in edges:
    if points[e].min() > 0:
      cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                    (points[e[1], 0], points[e[1], 1]), c, 2)
  return img

def send_pose_to_tf(tran_x, tran_y, tran_z, edge, br):
    x = tran_x / 10.0
    y = tran_y / 10.0
    z = tran_z / 10.0
    roll = 0
    pitch = 0
    yaw = 0

    br.sendTransform((x, y, z), tf.transformations.quaternion_from_euler(roll, pitch, yaw),
                     rospy.Time.now(),
                     joints[edge[0]],
                     joints[edge[1]])

    print("base:{0}, sub_base:{1}".format(joints[edge[1]], joints[edge[0]]))


def pose_rviz(points_3d):
    edges = mpii_edges

    points_3d = points_3d.reshape((-1, 16, 3))

    # Pelvic joint as a base point
    for num in range(points_3d.shape[0]):
        points_3d[num] = points_3d[num] - points_3d[num, 6:7]

    rospy.init_node("py_tf_broadcaster")
    print("发布~")
    br = tf.TransformBroadcaster()

    if not rospy.is_shutdown():

        for i in range(points_3d.shape[0]):
            point_3d = points_3d[i, :, :]

            X, Y, Z = np.zeros((3, points_3d.shape[1]))
            for j in range(point_3d.shape[0]):
                X[j] = 256 - point_3d[j, 0].copy()
                Y[j] = 256 - point_3d[j, 2].copy()
                Z[j] = 256 - point_3d[j, 1].copy()

            # adjust the coordinate of knee joint
            # if (Z[9] >= 0.) and (Z[1] < 0.):
            #     Z[1] = (Z[0] + Z[2]) / 2.0
            #
            # if (Z[9] >= 0.) and (Z[4] < 0.):
            #     Z[4] = (Z[3] + Z[5]) / 2.0

            for edge in edges:
                tran_x = X[edge[0]] - X[edge[1]]
                tran_y = Y[edge[0]] - Y[edge[1]]
                tran_z = Z[edge[0]] - Z[edge[1]]

                print("tran_x: ", tran_x)
                print("tran_y: ", tran_y)
                print("tran_z: ", tran_z)
                send_pose_to_tf(tran_x, tran_y, tran_z, edge, br)

            # rate.sleep()
            time.sleep(0.1)
