#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import tf

import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty, SetBool, SetBoolRequest  
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from robotics_project.srv import MoveHead, MoveHeadRequest, MoveHeadResponse
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from sensor_msgs.msg import JointState

from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry

from std_msgs.msg import String

from moveit_msgs.msg import MoveItErrorCodes
moveit_error_dict = {}
for name in MoveItErrorCodes.__dict__.keys():
    if not name[:1] == '_':
        code = MoveItErrorCodes.__dict__[name]
        moveit_error_dict[code] = name



class StateMachine(object):
    def __init__(self):
        
        self.node_name = "Student SM"

        # Access rosparams
        self.cmd_vel_top = rospy.get_param(rospy.get_name() + '/cmd_vel_topic')
        self.mv_head_srv_nm = rospy.get_param(rospy.get_name() + '/move_head_srv')
        self.pick_srv_nm = rospy.get_param(rospy.get_name()+'/pick_srv')
        self.place_srv_nm = rospy.get_param(rospy.get_name()+'/place_srv')

        

        #define cube pose message
        self.cube_mess = PoseStamped()
        # get cube pose parameters
        cube_pose_str = rospy.get_param(rospy.get_name()+'/cube_pose')
        cube_pose = [float(i) for i in cube_pose_str.split(',')]
        print(cube_pose)
        
        self.cube_mess.header.frame_id = 'base_footprint'
        self.cube_mess.pose.position.x = cube_pose[0]
        self.cube_mess.pose.position.y = cube_pose[1]
        self.cube_mess.pose.position.z = cube_pose[2]
        self.cube_mess.pose.orientation.x = cube_pose[3]
        self.cube_mess.pose.orientation.y = cube_pose[4]
        self.cube_mess.pose.orientation.z = cube_pose[5]
        self.cube_mess.pose.orientation.w = cube_pose[6]
        print(self.cube_mess)

        # pub = rospy.Publisher('/marker_pose_topic',PoseStamped,queue_size=10)
        # pub.publish(self.cube_mess)

        # Wait for service providers
        print(self.mv_head_srv_nm)
        print(self.pick_srv_nm)
        print(self.place_srv_nm)
        rospy.wait_for_service(self.mv_head_srv_nm, timeout=30)
        rospy.wait_for_service(self.pick_srv_nm,timeout=30)
        rospy.wait_for_service(self.place_srv_nm,timeout=30)
        

        # Instantiate publishers
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_top, Twist, queue_size=10)


        # Set up action clients
        rospy.loginfo("%s: Waiting for play_motion action server...", self.node_name)
        self.play_motion_ac = SimpleActionClient("/play_motion", PlayMotionAction)
        if not self.play_motion_ac.wait_for_server(rospy.Duration(1000)):
            rospy.logerr("%s: Could not connect to /play_motion action server", self.node_name)
            exit()
        rospy.loginfo("%s: Connected to play_motion action server", self.node_name)

        # Init state machine
        self.state = 1
        rospy.sleep(3)
        self.check_states()



    def check_states(self):

        while not rospy.is_shutdown() and self.state != 8:
            
            # State 0: Move the robot "manually" to door
            if self.state == 0:
                move_msg = Twist()
                move_msg.linear.x = 1

                rate = rospy.Rate(10)
                converged = False
                cnt = 0
                rospy.loginfo("%s: Moving towards door", self.node_name)
                while not rospy.is_shutdown() and cnt < 25:
                    self.cmd_vel_pub.publish(move_msg)
                    rate.sleep()
                    cnt = cnt + 1

                self.state = 1
                rospy.sleep(1)

            # State 1:  Tuck arm 
            if self.state == 1:
                rospy.loginfo("%s: Tucking the arm...", self.node_name)
                goal = PlayMotionGoal()
                goal.motion_name = 'home'#'pick_final_pose'
                goal.skip_planning = False
                #joint state info?
                # mess = rospy.wait_for_message("/joint_states", JointState)
                # print("my message: ")
                # print(mess)

                self.play_motion_ac.send_goal(goal)
                
                success_tucking = self.play_motion_ac.wait_for_result(rospy.Duration(100.0))
                if success_tucking:
                    rospy.loginfo("%s: Arm tucked.", self.node_name)
                    self.state = 3
                else:
                    self.play_motion_ac.cancel_goal()
                    rospy.logerr("%s: play_motion failed to tuck arm, reset simulation", self.node_name)
                    self.state = 5

                rospy.sleep(1)

            # State 2:  Move the robot "manually" to chair
            if self.state == 2:
                move_msg = Twist()
                move_msg.angular.z = 0

                rate = rospy.Rate(10)
                converged = False
                cnt = 0
                rospy.loginfo("%s: Moving towards table", self.node_name)
                while not rospy.is_shutdown() and cnt < 5:
                    self.cmd_vel_pub.publish(move_msg)
                    rate.sleep()
                    cnt = cnt + 0.8

                move_msg.linear.x = 0.5
                move_msg.angular.z = 0
                cnt = 0
                while not rospy.is_shutdown() and cnt < 15:
                    self.cmd_vel_pub.publish(move_msg)
                    rate.sleep()
                    cnt = cnt + 1

                self.state = 3
                rospy.sleep(1)

            # State 3:  Lower robot head service
            if self.state == 3:
            	try:
                    rospy.loginfo("%s: Lowering robot head", self.node_name)
                    move_head_srv = rospy.ServiceProxy(self.mv_head_srv_nm, MoveHead)
                    move_head_req = move_head_srv("down")
                    
                    if move_head_req.success == True:
                        self.state = 4
                        rospy.loginfo("%s: Move head down succeded!", self.node_name)
                    else:
                        rospy.loginfo("%s: Move head down failed!", self.node_name)
                        self.state = 5

                    rospy.sleep(3)

                except rospy.ServiceException, e:
                    print("Service call to move_head server failed: %s"%e)     
            
            # State 4: pick up the cube
            if self.state == 4:
                    try:
                        pub = rospy.Publisher('/marker_pose_topic',PoseStamped,queue_size=10)
                        # pub.publish(self.cube_mess)
                        count = 0
                        r = rospy.Rate(10) #10 Hz
                        while not rospy.is_shutdown():
                            # print(self.cube_mess)
                            pub.publish(self.cube_mess)
                            r.sleep()
                            if count>10:
                                break
                            count+=1
                        # # while not rospy.is_shutdown():
                        #     rospy.loginfo("%s: pick cube", self.node_name)
                        #     pick_srv = rospy.ServiceProxy(self.pick_srv_nm, SetBool)
                        #     print("pick_srv end")
                        #     pick_req = pick_srv(True)
                        #     if pick_req.success == True:
                        #         break
                        rospy.loginfo("%s: pick cube", self.node_name)
                        pick_srv = rospy.ServiceProxy(self.pick_srv_nm, SetBool)
                        print("finish pick_srv")
                        pick_req = pick_srv(True)

                    
                        if pick_req.success == True:
                            self.state = 6
                            rospy.loginfo("%s: pick up succeded!", self.node_name)
                        else:
                            rospy.loginfo("%s: pick up failed!", self.node_name)
                            self.state = 5
                        rospy.sleep(1)
                    except rospy.ServiceException,e:
                        print("Service call to pick up server failed: %s"%e)

            # move to second table
            if self.state == 6:
                print("this is state 6")
                move_msg = Twist()
                move_msg.angular.z = 4.0

                rate = rospy.Rate(10)
                converged = False
                cnt = 0.
                while not rospy.is_shutdown() and cnt < 5:
                    self.cmd_vel_pub.publish(move_msg)
                    print("move_msg: ")
                    print(move_msg)
                    rate.sleep()
                    cnt = cnt + 0.4

                cn = 0.
                move_msg.linear.x = 0.6
                move_msg.angular.z = 0
                # move_msg.linear.y = -1
                while not rospy.is_shutdown() and cnt < 15:
                    self.cmd_vel_pub.publish(move_msg)
                    print("move_msg: ")
                    print(move_msg)
                    rate.sleep()
                    cnt = cnt + 0.5                
                self.state = 7
                rospy.sleep(1)
            
                if self.state == 7:
                    try:
                        rospy.loginfo("%s: place cube",self.node_name)
                        place_srv = rospy.ServiceProxy(self.place_srv_nm,SetBool)
                        print("finish place_srv")
                        place_req = place_srv(True)
                        if place_req.success == True:
                            self.state = 8
                            rospy.loginfo("%s: place succeded!",self.node_name)
                        else:
                            rospy.loginfo("%s: place up failed!", self.node_name)
                            self.state = 5
                    except rospy.ServiceException,e:
                        print("Service call to place server failed: %s"%e)


            #Error handling
            if self.state == 5:
                rospy.logerr("%s: State machine failed. Check your code and try again!", self.node_name)
                return

        rospy.loginfo("%s: State machine finished!", self.node_name)
        return


# import py_trees as pt, py_trees_ros as ptr

# class BehaviourTree(ptr.trees.BehaviourTree):

# 	def __init__(self):

# 		rospy.loginfo("Initialising behaviour tree")

# 		# go to door until at door
# 		b0 = pt.composites.Selector(
# 			name="Go to door fallback", 
# 			children=[Counter(30, "At door?"), Go("Go to door!", 1, 0)]
# 		)

# 		# tuck the arm
# 		b1 = TuckArm()

# 		# go to table
# 		b2 = pt.composites.Selector(
# 			name="Go to table fallback",
# 			children=[Counter(5, "At table?"), Go("Go to table!", 0, -1)]
# 		)

# 		# move to chair
# 		b3 = pt.composites.Selector(
# 			name="Go to chair fallback",
# 			children=[Counter(13, "At chair?"), Go("Go to chair!", 1, 0)]
# 		)

# 		# lower head
# 		b4 = LowerHead()

# 		# become the tree
# 		tree = pt.composites.Sequence(name="Main sequence", children=[b0, b1, b2, b3, b4])
# 		super(BehaviourTree, self).__init__(tree)

# 		# execute the behaviour tree
# 		self.setup(timeout=10000)
# 		while not rospy.is_shutdown(): self.tick_tock(1)


# class Counter(pt.behaviour.Behaviour):

# 	def __init__(self, n, name):

# 		# counter
# 		self.i = 0
# 		self.n = n

# 		# become a behaviour
# 		super(Counter, self).__init__(name)

# 	def update(self):

# 		# count until n
# 		while self.i <= self.n:

# 			# increment count
# 			self.i += 1

# 			# return failure :(
# 			return pt.common.Status.FAILURE

# 		# succeed after counter done :)
# 		return pt.common.Status.SUCCESS


# class Go(pt.behaviour.Behaviour):

# 	def __init__(self, name, linear, angular):

# 		# action space
# 		self.cmd_vel_top = rospy.get_param(rospy.get_name() + '/cmd_vel_topic')
# 		self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_top, Twist, queue_size=10)

# 		# command
# 		self.move_msg = Twist()
# 		self.move_msg.linear.x = linear
# 		self.move_msg.angular.z = angular

# 		# become a behaviour
# 		super(Go, self).__init__(name)

# 	def update(self):

# 		# send the message
# 		rate = rospy.Rate(10)
# 		self.cmd_vel_pub.publish(self.move_msg)
# 		rate.sleep()

# 		# tell the tree that you're running
# 		return pt.common.Status.RUNNING


# class TuckArm(pt.behaviour.Behaviour):

# 	def __init__(self):

# 		# Set up action client
# 		self.play_motion_ac = SimpleActionClient("/play_motion", PlayMotionAction)

# 		# personal goal setting
# 		self.goal = PlayMotionGoal()
# 		self.goal.motion_name = 'home'
# 		self.goal.skip_planning = True

# 		# execution checker
# 		self.sent_goal = False
# 		self.finished = False

# 		# become a behaviour
# 		super(TuckArm, self).__init__("Tuck arm!")

# 	def update(self):

# 		# already tucked the arm
# 		if self.finished: 
# 			return pt.common.Status.SUCCESS
		
# 		# command to tuck arm if haven't already
# 		elif not self.sent_goal:

# 			# send the goal
# 			self.play_motion_ac.send_goal(self.goal)
# 			self.sent_goal = True

# 			# tell the tree you're running
# 			return pt.common.Status.RUNNING

# 		# if I was succesful! :)))))))))
# 		elif self.play_motion_ac.get_result():

# 			# than I'm finished!
# 			self.finished = True
# 			return pt.common.Status.SUCCESS

# 		# if I'm still trying :|
# 		else:
# 			return pt.common.Status.RUNNING
		


# class LowerHead(pt.behaviour.Behaviour):

# 	def __init__(self):

# 		# server
# 		mv_head_srv_nm = rospy.get_param(rospy.get_name() + '/move_head_srv')
# 		self.move_head_srv = rospy.ServiceProxy(mv_head_srv_nm, MoveHead)
# 		rospy.wait_for_service(mv_head_srv_nm, timeout=30)

# 		# execution checker
# 		self.tried = False
# 		self.tucked = False

# 		# become a behaviour
# 		super(LowerHead, self).__init__("Lower head!")

# 	def update(self):

# 		# try to tuck head if haven't already
# 		if not self.tried:

# 			# command
# 			self.move_head_req = self.move_head_srv("down")
# 			self.tried = True

# 			# tell the tree you're running
# 			return pt.common.Status.RUNNING

# 		# react to outcome
# 		else: return pt.common.Status.SUCCESS if self.move_head_req.success else pt.common.Status.FAILURE


	

if __name__ == "__main__":

	rospy.init_node('main_state_machine')
    # # init publisher
    # pub = rospy.Publisher('/robotics_intro/logic_state_machine',Twist,queue_size=10)

    # # init simple action client
    # manipulation_client = SimpleActionClient('manipulation_client',PlayMotionAction)

    # cuda_pose = rospy.get_param("/robotics_intro/logic_state_machine/cube_pose")
    # print(cuda_pose)

    
	try:
		#StateMachine()
		StateMachine()

    

	except rospy.ROSInterruptException:
		pass

	rospy.spin()
