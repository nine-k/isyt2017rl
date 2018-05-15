#!/usr/bin python

import rospy
from copy import deepcopy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from kobuki_msgs.msg import BumperEvent

import threading

import gym
import numpy
import random
import subprocess
import signal

position = None
lidar = None
collision = False

def update_position(data):
    global position
    position = (data.pose.pose.position.x, data.pose.pose.position.y)

def update_lidar(data):
    global lidar
    lidar = data.ranges

def set_collision(data):
    global collision
    print("Collided")
    collision = True

def ROS_LOOP(obj=None):
    rospy.init_node('LEARN', anonymous=False)
    rospy.Subscriber('/odom', Odometry, update_position, queue_size=1)
    rospy.Subscriber('/scan', LaserScan, update_lidar, queue_size=1)
    rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, set_collision, queue_size=1)
    while work:
        pass

class TurtleBotObstEnv(gym.Env):
    def __init__(self):
        self.goal_reward = None
        self.obst_punish = None
        self.cur_task_num = None
        self.cur_gazebo_proc = None
        self.cur_pos = None
        self.goal_pos = None
        self.goal_thresh = None
        self.goal = None
        #self.ros_thread = threading.Thread(target=ROS_lOOP, kwargs=dict(obj=self))
        self.turtle_bot_pub = None
        self.task_count = None
        self.ros_launch_path = None
        self.position = None
        self.lidar = None

    def _configure(self,
                    goal_reward=10,
                    obs_punish=5,
                    map_dir="/home/kolya/Documents/maps/train/",
                    goal_thresh=0.2,
                    task_count=12
                    ):
        self.goal_reward = goal_reward
        self.obst_punish = abs(obs_punish)
        self.cur_task_num = 0
        self.goal_thresh = goal_thresh
        self.task_count = task_count
        self.map_dir = map_dir
        self.goal = (1, 1)
        self.ros_launch_path = "/opt/ros/kinetic/bin/roslaunch"
        rospy.init_node('LEARN', anonymous=False)
        #rospy.Subscriber('/odom', Odometry, update_position, queue_size=1)
        #rospy.Subscriber('/scan', LaserScan, update_lidar, queue_size=1)
        #rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, set_collision, queue_size=1)
        self.turtle_bot_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

    def update_posisition(self):
        data = rospy.wait_for_message('/odom', Odometry, timeout=15)
        self.position = (data.pose.pose.position.x, data.pose.pose.position.y)

    def update_lidar(self):
        data = rospy.wait_for_message('/scan', LaserScan, timeout=15)
        self.lidar = data.ranges

    def check_collision(self):
        try:
            rospy.wait_for_message('/mobile_base/events/bumper', BumperEvent, timeout=0.2)
            return True
        except:
            return False

    def __repr__(self):
        return self.__class__.__name__

    def _reset(self):
        self.cur_task_num = (self.cur_task_num + 1) % self.task_count
        if self.cur_gazebo_proc is not None:
            #self.cur_gazebo_proc.kill()
            self.cur_gazebo_proc.send_signal(signal.SIGINT)
            try:
                print("waiting for gazebo to die")
                self.cur_gazebo_proc.communicate()
                print("KILLED")
            except:
                print("no gazebo to kill")
        map_dir = self.map_dir + '/'+ str(self.cur_task_num)
        print(self.ros_launch_path + " turtlebot_gazebo turtlebot_world.launch world_file:=" + map_dir)
        self.cur_gazebo_proc = subprocess.Popen([self.ros_launch_path, "turtlebot_gazebo", "turtlebot_world.launch", "world_file:=" + map_dir], shell=False)
        try:
            self.update_lidar()
        except:
            print("gazebo not responding")
        self.update_position()
        #TODO WRITE SET POSITION
        return self._init_state()

    def _seed(self):
        pass

    def _init_state(self):
        return self._get_state()

    def _get_base_state(self, cur_position_discrete):
        #get angle to target and draw a verticall line
        #remove nan values
        return self.lidar

    def _get_state(self):
        return self.lidar

    def write_speed(self, action):
        msg = Twist()
        msg.linear.x = action[0]
        msg.angular.z = action[1]
        self.turtle_bot_pub.publish(msg)
        return


    def _step(self, action):
        cur_pos = self.position
        dist_to_goal = ((cur_pos[0] - self.goal[0])**2 + (cur_pos[1] - self.goal[1])**2)**0.5
        done = dist_to_goal < self.goal_thresh
        reward = 0
        if (done):
            reward = self.goal_reward
        elif self.check_collision():
            reward = -self.obst_punish
        self.write_speed(action)
        obs = self._get_state()
        return obs, reward, done, None

