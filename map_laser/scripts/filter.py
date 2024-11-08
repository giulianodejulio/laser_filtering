#!/usr/bin/env python

import rospy
import tf
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion, quaternion_matrix
import math
import numpy as np


class MapLaser(object):
    def __init__(self):
        rospy.init_node('map_laser_filter')
        mobile_base = rospy.get_param("~mobile_base")
        self.pub = rospy.Publisher('/base_scan_filter',
                                   LaserScan,
                                   queue_size=10)
        self.listener = tf.TransformListener()
        self.map = None
        self.save = None
        self.scan_sub = rospy.Subscriber('/scan_filtered',
                                    LaserScan,
                                    self.laser_cb,
                                    queue_size=1)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_cb)
        self.global_frame = mobile_base + "_map"

        # Angular limits in radians
        self.min_angle = math.radians(-135)  # -135 degrees
        self.max_angle = math.radians(45)    # 45 degrees
        self.max_distance = rospy.get_param("~laser_scan_max_distance_range") # max distance range of laser scan. Set equal to obstacle_range ROS parameter in legtracker_lstm_nodes.launch

    def map_cb(self, msg):
        self.map = msg

    def get_laser_frame(self, msg):
        now = msg.header.stamp
        laser_frame = msg.header.frame_id
        self.listener.waitForTransform(self.global_frame, laser_frame, now, rospy.Duration(.15))

        return self.listener.lookupTransform(self.global_frame, laser_frame, now)

    def is_occupied(self, x, y):
        N = 4
        for dx in range(-N, N + 1):
            for dy in range(-N, N + 1):
                index = (x + dx) + (y + dy) * self.map.info.width
                if index < 0 or index > len(self.map.data):
                    continue
                value = self.map.data[index]
                if value > 50 or value < 0:
                    return True
        return False

    def laser_cb(self, msg):
        if self.map is None:
            return
        try:
            (trans, rot) = self.get_laser_frame(msg)
            self.save = (trans, rot)
        except tf.Exception:
            if self.save is None:
                return
            (trans, rot) = self.save

        yaw = euler_from_quaternion(rot)[2]  # orientation of laser frame x-axis wrt to global frame x-axis

        nr = []

        for (i, d) in enumerate(msg.ranges):
            if math.isnan(d) or d > msg.range_max or d < msg.range_min:
                nr.append(msg.range_max + 1.0)
                continue

            # Calculate the angle relative to the laser frame
            angle_local = msg.angle_min + msg.angle_increment * i

            # Check if the angle is within the desired range (-135° to 45°)
            if angle_local < self.min_angle or angle_local > self.max_angle:
                nr.append(msg.range_max + 1.0)  # Out of angular bounds, ignore
                continue

            # Limit the range to a maximum of self.max_distance meters
            if d > self.max_distance:
                nr.append(msg.range_max + 1.0)  # Out of distance bounds, ignore
                continue

            # Compute the global coordinates of the laser point
            angle_global = yaw - angle_local  # laser frame has z axis down, global frame has z axis up
            dx_local = math.cos(angle_local) * d
            dy_local = math.sin(angle_local) * d
            
            transform_matrix = quaternion_matrix(rot)
            rotation_matrix_xy = transform_matrix[:2, :2]
            [dx_global, dy_global] = np.dot(rotation_matrix_xy, [dx_local, dy_local])

            map_x = trans[0] + dx_global
            map_y = trans[1] + dy_global

            grid_x = int((map_x - self.map.info.origin.position.x) / self.map.info.resolution)
            grid_y = int((map_y - self.map.info.origin.position.y) / self.map.info.resolution)

            if self.is_occupied(grid_x, grid_y):
                nr.append(msg.range_max + 1.0)
            else:
                nr.append(d)

        msg.ranges = nr
        self.pub.publish(msg)


if __name__ == '__main__':
    x = MapLaser()
    rospy.spin()
