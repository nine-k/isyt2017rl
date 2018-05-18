#!/usr/bin python

import rospy
from copy import deepcopy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from kobuki_msgs.msg import BumperEvent
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates


import gym
import numpy
import random
import subprocess
import signal
import os
import math
import random
from time import sleep
from tf.transformations import euler_from_quaternion
#from collections import namedtuple
from recordclass import recordclass

position = None
lidar = None
collision = False

debug_info = recordclass('extra_info', "dist collisions goal".split())

def update_position(data):
    global position
    position = (data.pose.pose.position.x, data.pose.pose.position.y)

def update_lidar(data):
    global lidar
    lidar = data.ranges

def set_collision(data):
    global collision
    #print("Collided")
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
        self.idle_punish = None
        self.idle_thresh = None
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
        self.visualize = None
        self.prev_pos = None
        self.default_lidar_val = None
        self.reset_proxy = None
        self.cur_map_it = None
        self.its_per_map = None
        self.scan_min_ang = -0.52156
        self.scan_max_ang = 0.52427
        self.ang_delta = 0.00163
        self.cur_angle = 0
        self.extra_info = debug_info(0,0,0)

    def point_dist(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    def clean_gazebo(self):
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        rbsp = tmp.count('robot_state_publisher')
        ndlt = tmp.count('nodelet')
        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rbsp > 0:
            os.system("killall -9 robot_state_publisher")
        if ndlt > 0:
            os.system("killall -9 nodelet")

        sleep(3.5)
        #tmp = os.popen("ps -Af").read()
        #gzclient_count = tmp.count('gzclient')
        #gzserver_count = tmp.count('gzserver')
        #if (gzclient_count > 0 or gzserver_count > 0):
        #    try:
        #        os.wait()
        #    except:
        #        pass

    def _configure(self,
                    goal_reward=20,
                    obs_punish=5,
                    idle_punish=2,
                    map_dir="/home/kolya/Documents/maps/train/",
                    goal_thresh=0.5,
                    task_count=11,
                    idle_thresh=0.01,
                    visualize=True,
                    def_lid_val=0,
                    its_per_map=100
                    ):
        self.goal_reward = goal_reward
        self.idle_thresh = idle_thresh
        self.idle_punish = abs(idle_punish)
        self.obst_punish = abs(obs_punish)
        self.cur_task_num = 0
        self.goal_thresh = goal_thresh
        self.task_count = task_count
        self.map_dir = map_dir
        self.goal = (1, 1)
        #self.ros_launch_path = "/opt/ros/kinetic/bin/roslaunch"
        self.ros_launch_path = "roslaunch"
        self.visualize = visualize
        self.default_lidar_val = def_lid_val
        rospy.init_node('LEARN', anonymous=False)
        #rospy.Subscriber('/odom', Odometry, update_position, queue_size=1)
        #rospy.Subscriber('/scan', LaserScan, update_lidar, queue_size=1)
        rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, set_collision, queue_size=1)
        self.turtle_bot_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.its_per_map = its_per_map
        self.cur_map_it = its_per_map
        self.clean_gazebo()

    def quat_to_tuple(self, q):
        return (q.x, q.y, q.z, q.w)

    def update_position(self):
        counter = 0
        while True:
            try:
                data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5)
                break
            except:
                counter += 1
                if counter >= 2:
                    raise
                #print("waiting for pos readings")
        idx = data.name.index("mobile_base")
        data = data.pose[idx]
        self.cur_angle = euler_from_quaternion(self.quat_to_tuple(data.orientation))[2]
        #print(self.cur_angle)
        #data = rospy.wait_for_message('/odom', Odometry, timeout=15)
        if self.prev_pos is not None:
            self.prev_pos = (self.position[0], self.position[1])
            self.position = (data.position.x, data.position.y)
        else:
            self.position = (data.position.x, data.position.y)
            self.prev_pos = (self.position[0], self.position[1])

    def update_lidar(self):
	#angle_min: -0.521567881107rad = -29 deg
	#angle_max: 0.524276316166rad = -30 deg
	#angle_increment: 0.00163668883033
	#time_increment: 0.0
	#scan_time: 0.0329999998212
	#range_min: 0.449999988079
	#range_max: 10.0
        counter = 0
        while True:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                break
            except:
                counter += 1
                if counter >= 2:
                    raise
                #print("waiting for scan readings")
        self.lidar = list(data.ranges)
        for i in range(len(self.lidar)):
            if math.isnan(self.lidar[i]):
                self.lidar[i] = self.default_lidar_val

    def check_collision(self):
        global collision
        if collision:
            collision = False
            return True
        return False
        #try:
        #    rospy.wait_for_message('/mobile_base/events/bumper', BumperEvent, timeout=0.2)
        #    return True
        #except:
        #    return False

    def __repr__(self):
        return self.__class__.__name__

    def calc_angle_to_goal(self):
        dist = self.point_dist(self.position, self.goal)
        phi = math.acos((self.goal[0] - self.position[0]) / dist)
        if (self.goal[1] - self.position[1]) < 0:
            phi *= -1
        #print("phi:", phi)
        return phi

    def _reset(self):
        self.cur_task_num = (self.cur_task_num % self.task_count) + 1
        while True:
            if self.cur_gazebo_proc is not None:
                self.cur_gazebo_proc.kill()
                self.cur_gazebo_proc.send_signal(signal.SIGINT)
                sleep(1.5)
            self.clean_gazebo()

            map_dir = self.map_dir + '/'+ str(self.cur_task_num)
            gui = "gui:=false"
            if self.visualize:
                gui = "gui:=true"
            #print(self.ros_launch_path + " turtlebot_gazebo turtlebot_world.launch world_file:=" + map_dir + ' ' + gui)
            self.cur_gazebo_proc = subprocess.Popen([self.ros_launch_path, "turtlebot_gazebo", "turtlebot_world.launch", "world_file:=" + map_dir, gui], shell=False)
            self.cur_map_it = 0
            try:
                self.update_lidar()
                self.update_position()
                break
            except:
                print("restarting ros cuz of error")
        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause()

        self.extra_info.goal = self.goal
        self.extra_info.collisions = 0
        self.goal = (random.uniform(-4, 4), random.uniform(-4, 4))
        return self._init_state()

    def _seed(self):
        pass

    def _init_state(self):
        return self._get_state()

    def _get_base_state(self, cur_position_discrete):
        return self.lidar

    def _get_state(self):
        #get angle to target and draw a verticall line
        #self.lidar
        phi = self.cur_angle - self.calc_angle_to_goal()
        if phi < -math.pi:
            phi += 2*math.pi
        if phi > math.pi:
            phi -= 2*math.pi
        phi = min(self.scan_max_ang, phi)
        phi = max(self.scan_min_ang, phi)
        idx = int(phi / self.ang_delta)
        idx += len(self.lidar) / 2
        idx = max(0, idx)
        idx = min(len(self.lidar) - 1, idx)
        #print(phi, idx)
        self.lidar[idx] = -1 * self.point_dist(self.goal, self.position)
        return self.lidar

    def write_speed(self, action):
        msg = Twist()
        msg.linear.x = action[0]
        msg.angular.z = action[1]
        self.turtle_bot_pub.publish(msg)
        return


    def _step(self, action):
        if (action == 0):
            action = (0.5, 0)
        elif (action == 1):
            action = (0, -0.3)
        elif (action == 2):
            action = (0, 0.3)
        elif (action == 3):
            action = (-0.25, 0)

        try:
            rospy.wait_for_service('/gazebo/unpause_physics')
            self.unpause()

            self.write_speed(action)
            self.update_lidar()
            self.update_position()
            rospy.wait_for_service('/gazebo/pause_physics')
            self.pause()
        except:
            return self._get_state(), 0, True, self.extra_info

        cur_pos = self.position
        dist_to_goal = self.point_dist(cur_pos, self.goal)
        dist_gain = (self.point_dist(self.goal, self.prev_pos) - dist_to_goal) * 0.0001
        #dist_to_goal = ((cur_pos[0] - self.goal[0])**2 + (cur_pos[1] - self.goal[1])**2)**0.5
        done = dist_to_goal < self.goal_thresh
        reward = dist_gain
        if (done):
            reward = self.goal_reward
        elif self.check_collision():
            reward += -self.obst_punish
            self.extra_info.collisions += 1
        elif self.point_dist(cur_pos, self.prev_pos) < self.idle_thresh and abs(action[1]) < 0.1:
            reward += -self.idle_punish
        else:
            reward += -1
        obs = self._get_state()
        #print(self.position, self.goal)
        #print("dist:", dist_to_goal)
        self.extra_info.dist = dist_to_goal
        return obs, reward, done, self.extra_info

