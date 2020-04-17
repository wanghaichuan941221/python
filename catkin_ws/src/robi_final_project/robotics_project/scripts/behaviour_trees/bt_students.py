#!/usr/bin/env python

import py_trees as pt, py_trees_ros as ptr, rospy
from behaviours_student import *
from reactive_sequence import RSequence

import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty, SetBool, SetBoolRequest  
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped,PoseArray
from robotics_project.srv import MoveHead, MoveHeadRequest, MoveHeadResponse
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from sensor_msgs.msg import JointState

from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry

from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from actionlib_msgs.msg import GoalID

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from gazebo_msgs.srv import DeleteModel,SpawnModel
from geometry_msgs.msg import Pose,Point,Quaternion
from gazebo_msgs.srv import SetModelState


goToPick_tried = False
pickCube_tried = False
goToPlace_tried = False
placeCube_tried = False
move_head_up1 = False
move_head_up2 = False
move_head_down1 = False
move_head_down2 = False
go_Back = False
tuck_arm = False
spawn_cube = False
move_robot = False


class spawnGazeboModel(pt.behaviour.Behaviour):
    def __init__(self):
        # self.del_model_prox = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel) # Handle to model spawner
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.aruco_cube = rospy.get_param(rospy.get_name() + '/aruco_cube_sdf')
        # self.set_model = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        self.tried = False
        self.done = False

        super(spawnGazeboModel, self).__init__("spawnGazeboModel")
    
    def update(self):
        global spawn_cube 
        if spawn_cube == False:
            self.done = False
            self.tried = False

        model_name = 'new_aruco_cube_3'

        if self.done:
            print("spawn cube self done")
            return pt.common.Status.SUCCESS
        elif not self.tried:
            print("spawn cube self tried")
            cube_pose = Pose(Point(-1.130530,-6.653650,0.86250),Quaternion(0.0,0.0,0.0,1.0))
    
            f = open(self.aruco_cube,'r')
            sdffile = f.read()
            spawn_model_srv  = self.spawn_model(model_name, sdffile, "/", cube_pose, "map")
            if spawn_model_srv.success:
                self.done = True
                rospy.sleep(1)
                return pt.common.Status.SUCCESS
        if (not spawn_model_srv.success):
            return pt.common.Status.FAILURE
        # if still trying
        else:
            return pt.common.Status.RUNNING


class tuckarm1(pt.behaviour.Behaviour):

    """
    Sends a goal to the tuck arm action server.
    Returns running whilst awaiting the result,
    success if the action was succesful, and v.v..
    """

    def __init__(self):

        rospy.loginfo("Initialising tuck arm behaviour.")

        # Set up action client
        self.play_motion_ac = SimpleActionClient("/play_motion", PlayMotionAction)

        # personal goal setting
        self.goal = PlayMotionGoal()
        self.goal.motion_name = 'home'
        self.goal.skip_planning = True

        # execution checker
        self.sent_goal = False
        self.finished = False

        # become a behaviour
        super(tuckarm1, self).__init__("Tuck arm!")

    def update(self):
        global tuck_arm
        if tuck_arm == False:
            self.finished = False
            self.sent_goal = False

        # already tucked the arm
        if self.finished:
            print("tuck arm self finished") 
            return pt.common.Status.SUCCESS
        
        # command to tuck arm if haven't already
        elif not self.sent_goal:
            
            # send the goal
            self.play_motion_ac.send_goal(self.goal)
            print("tuck arm self send goal")
            if self.play_motion_ac.get_result():
                print("tuck arm inside sucess loop")
                # than I'm finished!
                # rospy.sleep(2)
                tuck_arm = True
                self.finished = True
                return pt.common.Status.SUCCESS
            self.sent_goal = True
            # tell the tree you're running
            return pt.common.Status.RUNNING

        # if I was succesful! :)))))))))
        # elif self.play_motion_ac.get_result():

        #     # than I'm finished!
        #     self.finished = True
        #     return pt.common.Status.SUCCESS

        # if failed
        elif not self.play_motion_ac.get_result():
            return pt.common.Status.FAILURE

        # if I'm still trying :|
        else:
            return pt.common.Status.RUNNING

class move_headup1(pt.behaviour.Behaviour):

    def __init__(self):

        rospy.loginfo("Initialising move head behaviour.")

        # server
        mv_head_srv_nm = rospy.get_param(rospy.get_name() + '/move_head_srv')
        self.move_head_srv = rospy.ServiceProxy(mv_head_srv_nm, MoveHead)
        rospy.wait_for_service(mv_head_srv_nm, timeout=30)

        # head movement direction; "down" or "up"
        self.direction = "up"

        # execution checker
        self.tried = False
        self.done = False

        # become a behaviour
        super(move_headup1, self).__init__("Lower head!")

    def update(self):
        global move_head_up1
        rospy.sleep(2)
        if move_head_up1 == False:
            self.done = False
            self.tried = False
        # success if done
        if self.done:
            print("move head up1 self done")
            return pt.common.Status.SUCCESS

        # try if not tried
        elif not self.tried:

            # command
            self.move_head_req = self.move_head_srv(self.direction)

            if self.move_head_req.success:
                move_head_up1 = True
                self.done = True
                return pt.common.Status.SUCCESS

            self.tried = True
            # tell the tree you're running
            return pt.common.Status.RUNNING

        # if succesful
        # elif self.move_head_req.success:
        #     self.done = True
        #     return pt.common.Status.SUCCESS

        # if failed
        elif not self.move_head_req.success:
            return pt.common.Status.FAILURE

        # if still trying
        else:
            return pt.common.Status.RUNNING

class move_headup2(pt.behaviour.Behaviour):

    def __init__(self):

        rospy.loginfo("Initialising move head behaviour.")

        # server
        mv_head_srv_nm = rospy.get_param(rospy.get_name() + '/move_head_srv')
        self.move_head_srv = rospy.ServiceProxy(mv_head_srv_nm, MoveHead)
        rospy.wait_for_service(mv_head_srv_nm, timeout=30)

        # head movement direction; "down" or "up"
        self.direction = "up"

        # execution checker
        self.tried = False
        self.done = False

        # become a behaviour
        super(move_headup2, self).__init__("Lower head!")

    def update(self):
        global move_head_up2
        if move_head_up2 == False:
            self.done = False
            self.tried = False
        # success if done
        if self.done:
            print("move head up self done")
            return pt.common.Status.SUCCESS

        # try if not tried
        elif not self.tried:

            # command
            self.move_head_req = self.move_head_srv(self.direction)

            if self.move_head_req.success:
                move_head_up2 = True
                self.done = True
                return pt.common.Status.SUCCESS
                
            self.tried = True
            # tell the tree you're running
            return pt.common.Status.RUNNING

        # if succesful
        # elif self.move_head_req.success:
        #     self.done = True
        #     return pt.common.Status.SUCCESS

        # if failed
        elif not self.move_head_req.success:
            return pt.common.Status.FAILURE

        # if still trying
        else:
            return pt.common.Status.RUNNING

class move_headdown1(pt.behaviour.Behaviour):

    def __init__(self):

        rospy.loginfo("Initialising move head behaviour.")

        # server
        mv_head_srv_nm = rospy.get_param(rospy.get_name() + '/move_head_srv')
        self.move_head_srv = rospy.ServiceProxy(mv_head_srv_nm, MoveHead)
        rospy.wait_for_service(mv_head_srv_nm, timeout=30)

        # head movement direction; "down" or "up"
        self.direction = "down"

        # execution checker
        self.tried = False
        self.done = False

        # become a behaviour
        super(move_headdown1, self).__init__("Lower head!")

    def update(self):
        global move_head_down1
        if move_head_down1 == False:
            self.done = False
            self.tried = False
        # success if done
        if self.done:
            print("move head down 1")
            return pt.common.Status.SUCCESS

        # try if not tried
        elif not self.tried:

            # command
            self.move_head_req = self.move_head_srv(self.direction)

            if self.move_head_req.success:
                move_head_down1 = True
                self.done = True
                return pt.common.Status.SUCCESS
                
            self.tried = True
            # tell the tree you're running
            return pt.common.Status.RUNNING

        # if succesful
        # elif self.move_head_req.success:
        #     self.done = True
        #     return pt.common.Status.SUCCESS

        # if failed
        elif not self.move_head_req.success:
            return pt.common.Status.FAILURE

        # if still trying
        else:
            return pt.common.Status.RUNNING



class move_headdown2(pt.behaviour.Behaviour):

    def __init__(self):

        rospy.loginfo("Initialising move head behaviour.")

        # server
        mv_head_srv_nm = rospy.get_param(rospy.get_name() + '/move_head_srv')
        self.move_head_srv = rospy.ServiceProxy(mv_head_srv_nm, MoveHead)
        rospy.wait_for_service(mv_head_srv_nm, timeout=30)

        # head movement direction; "down" or "up"
        self.direction = "down"

        # execution checker
        self.tried = False
        self.done = False

        # become a behaviour
        super(move_headdown2, self).__init__("Lower head!")

    def update(self):
        global move_head_down2
        rospy.sleep(1)
        if move_head_down2 == False:
            self.done = False
            self.tried = False
        # success if done
        if self.done:
            print("move head down2 self done")
            return pt.common.Status.SUCCESS

        # try if not tried
        elif not self.tried:
            print("move head down2 not self.tried")
            # command           
               
            self.move_head_req = self.move_head_srv(self.direction)

            if self.move_head_req.success:
                move_head_down2 = True
                self.done = True
                return pt.common.Status.SUCCESS
            elif not self.move_head_req.success:
                return pt.common.Status.FAILURE
                
            self.tried = True
            # tell the tree you're running
            return pt.common.Status.RUNNING

        # if succesful
        # elif self.move_head_req.success:
        #     self.done = True
        #     return pt.common.Status.SUCCESS

        # if failed
        # elif not self.move_head_req.success:
        #     return pt.common.Status.FAILURE

        # if still trying
        else:
            return pt.common.Status.RUNNING


class pick_cube(pt.behaviour.Behaviour):
    def __init__(self):

        rospy.loginfo("Initialising pick cube behaviour.")
		
		# rospy.Subscriber("/detected_aruco_pose",PoseStamped,callback=self.callback)
		
        # server
        pick_srv_nm = rospy.get_param(rospy.get_name() + '/pick_srv')
        self.pick_srv = rospy.ServiceProxy(pick_srv_nm, SetBool)
        rospy.wait_for_service(pick_srv_nm, timeout=30)

        # # head movement direction; "down" or "up"
        # self.direction = direction

        # execution checker
        self.tried = False
        self.done = False

        # become a behaviour
        super(pick_cube, self).__init__("pick cube!")



    def update(self):
        global pickCube_tried
        if pickCube_tried == False:
            self.done = False
            self.tried = False

        # success if done
        if self.done:
            print("pick cube self done")
            return pt.common.Status.SUCCESS

        # try if not tried
        elif not self.tried:

            # command
            self.pick_req = self.pick_srv(True)
            if self.pick_req.success:
                pickCube_tried = True
                self.done = True
                return pt.common.Status.SUCCESS

            self.tried = True
            # tell the tree you're running
            return pt.common.Status.RUNNING

        # if succesful
        # elif self.pick_req.success:
        #     pickCube_tried = True
        #     self.done = True
        #     return pt.common.Status.SUCCESS

        # if failed
        elif not self.pick_req.success:
            return pt.common.Status.FAILURE

        # if still trying
        else:
            return pt.common.Status.RUNNING

class place_cube(pt.behaviour.Behaviour):
    def __init__(self):

        rospy.loginfo("Initialising place cube behaviour.")
		
        # server
        place_srv_nm = rospy.get_param(rospy.get_name() + '/place_srv')
        self.place_srv = rospy.ServiceProxy(place_srv_nm, SetBool)
        rospy.wait_for_service(place_srv_nm, timeout=30)

        # # head movement direction; "down" or "up"
        # self.direction = direction

        # execution checker
        self.tried = False
        self.done = False

        # become a behaviour
        super(place_cube, self).__init__("place cube!")



    def update(self):
        global placeCube_tried
        if placeCube_tried == False:
            self.done = False
            self.tried = False

        # success if done
        if self.done:
            print("place cube self done")
            return pt.common.Status.SUCCESS

        # try if not tried
        elif not self.tried:
            # command
            self.place_req = self.place_srv(True)
            if self.place_req.success:
                placeCube_tried = True
                self.done = True
                return pt.common.Status.SUCCESS

            # self.tried = True
            self.tried = True

            # tell the tree you're running
            return pt.common.Status.RUNNING

        # if failed
        elif not self.place_req.success:
            return pt.common.Status.FAILURE 

        # if still trying
        else:
            return pt.common.Status.RUNNING


class detect_cube_on_second_table(pt.behaviour.Behaviour):
    def __init__(self):

        rospy.loginfo("Initialising detect cube on second table behaviour.")
		
        # server
        self.data_old = PoseStamped()
        self.data = PoseStamped()
        # rospy.Subscriber("/detected_aruco_pose",PoseStamped,callback=self.getMessage)

        # server
        # move_head_srv_nm = rospy.get_param(rospy.get_name() + '/move_head_srv')
        # self.pick_srv = rospy.ServiceProxy(move_head_srv_nm, SetBool)
        # rospy.wait_for_service(move_head_srv_nm, timeout=30)

        # # head movement direction; "down" or "up"
        # self.direction = direction

        # execution checker
        self.tried = False
        self.done = False

        # become a behaviour
        super(detect_cube_on_second_table, self).__init__("detect_cube_on_second_table!")

    def getMessage(self,data):
        self.data_old = self.data
        self.data = data
        print("my data")
        print(self.data_old)
        print(data)


    def isAngleZero(self,data):
        if data.pose.orientation.x==0.0 and data.pose.orientation.y==0.0 and data.pose.orientation.z == 0.0 and data.pose.orientation.w==1.0:
            return True
        return False


    def update(self):
        global pickCube_tried,placeCube_tried,goToPick_tried,goToPlace_tried,move_head_down1,move_head_down2,move_head_up1,move_head_up2,tuck_arm,spawn_cube


        rospy.sleep(3)
        print("detect cube")
        rospy.Subscriber('/detected_aruco_pose',PoseStamped,callback=self.getMessage)
        
        print("detect cube")


        if self.done:
            print("detect cube self done")
            return pt.common.Status.FAILURE

        # try if not tried
        elif not self.tried:
            print("detect cube not tried")
            self.tried = True
            return pt.common.Status.RUNNING

        elif self.isAngleZero(self.data)==False or self.isAngleZero(self.data_old)==False:
            print("detect cube failure")
            self.done = True
            return pt.common.Status.FAILURE

        # if failed
        # elif not self.place_req.success:
        elif self.isAngleZero(self.data)==True and self.isAngleZero(self.data_old)==True:
            pickCube_tried = False
            placeCube_tried = False
            goToPick_tried = False
            goToPlace_tried = False
            move_head_down1 = False
            move_head_down2 = False
            move_head_up1 = False
            move_head_up2 = False
            tuck_arm = False
            spawn_cube = False
            print("can not detect cube, will go back to table")
            return pt.common.Status.SUCCESS

        else:
            print("detect cube else")
            return pt.common.Status.RUNNING


class robotLocalization(pt.behaviour.Behaviour):

    def __init__(self):

        rospy.loginfo("Initialising localization behaviour.")
        
        # init the pick data to zeros
       
        # self.robot_old_pose = PoseWithCovarianceStamped()
        # self.robot_new_pose = PoseWithCovarianceStamped()
        # self.diff = 0



        self.cmd_vel_top = "/key_vel"
        rospy.loginfo(self.cmd_vel_top)
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_top, Twist, queue_size=10)

        # command
        self.move_msg = Twist()
        self.move_msg.linear.x = 0
        self.move_msg.angular.z = 80


        # execution checker
        self.tried = False
        self.done = False

        # become a behaviour
        super(robotLocalization, self).__init__("robotLocalization!")

    # def getCurrentRobotPose(self,data):
    #     # print("data0 pose")
    #     # print(data.pose.pose.position)
    #     # print("data1 pose")
    #     # print(data.pose)
    #     self.robot_old_pose = self.robot_new_pose
    #     self.robot_new_pose = data

    # def diffBetweenPose(self,new_pose,old_pose):
    #     diff = (new_pose.pose.pose.position.x - old_pose.pose.pose.position.x)**2 + (new_pose.pose.pose.position.y - old_pose.pose.pose.position.y)**2
        
    #     return diff


    def update(self):
        global move_robot
        if move_robot==True:
            self.tried = False
        # rospy.Subscriber('/amcl_pose',PoseWithCovarianceStamped,self.getCurrentRobotPose)
        # self.diff = self.diffBetweenPose(self.robot_new_pose,self.robot_old_pose)
        # print(self.diff)
        # rospy.sleep(1)
        # diff = self.diff
        # print(diff)

        # success if done
        # if self.done:
        #     # print("robot localization self done")
        #     return pt.common.Status.SUCCESS

        # try if not tried
        # el
        if not self.tried:
            print("robot localization not self.tried")
            self.local_srv = rospy.ServiceProxy('/global_localization', Empty)
            rospy.wait_for_service('/global_localization', timeout=30)
            self.local_srv()
            rate = rospy.Rate(10)
            count = 0
            while count<10:
                # print("publish message")
                # print(self.move_msg)
                self.cmd_vel_pub.publish(self.move_msg)
                count+=1
                rate.sleep()
            
            self.tried = True
            rospy.sleep(5)

            # tell the tree you're running
            return pt.common.Status.RUNNING

        # if succesful
        else:
            return pt.common.Status.SUCCESS

        # # if failed
        # elif not self.req.SUCCESS:
        #     return pt.common.Status.FAILURE

        # # if still trying
        # else:
        #     return pt.common.Status.RUNNING


class goToPickPose(pt.behaviour.Behaviour):

    def __init__(self):
        rospy.loginfo("Initialising place cube behaviour.")
        
        # init the pick data to zeros
        self.pick_data = PoseStamped()
        
        # execution checker
        self.tried = False
        self.done = False
        self.state = 0

        self.robot_new_pose = PoseWithCovarianceStamped()
        self.robot_old_pose = PoseWithCovarianceStamped()

        # become a behaviour
        super(goToPickPose, self).__init__("goToPickPose!")

    def getPickPoseMessage(self,data):
        # print("this is the pick data")
        # print(data)
        self.pick_data = data

    def getCurrentRobotPose(self,data):
        global move_robot
        self.robot_old_pose = self.robot_new_pose
        self.robot_new_pose = data
        if self.diffBetweenPose(self.robot_new_pose,self.robot_old_pose)>0.5:
            move_robot = False


    def diffBetweenPose(self,new_pose,old_pose):
        diff = (new_pose.pose.pose.position.x - old_pose.pose.pose.position.x)**2 + (new_pose.pose.pose.position.y - old_pose.pose.pose.position.y)**2
        
        return diff


    def update(self):
        global goToPick_tried,move_robot

        rospy.Subscriber('/amcl_pose',PoseWithCovarianceStamped,self.getCurrentRobotPose)
        # diff = self.diffBetweenPose(self.robot_new_pose,self.robot_old_pose)
        # print(diff)
        # if diff>0.5:
        #     move_robot = True

        if goToPick_tried == False:
            self.done = False
            self.tried = False

        rospy.Subscriber('/pick_pose',PoseStamped,callback=self.getPickPoseMessage)
        action_client = SimpleActionClient("/move_base",MoveBaseAction)
        action_client.wait_for_server()
        if self.done:
            print("go to pick self done")
            return pt.common.Status.SUCCESS

        elif not self.tried:
            print("inside goToPick_tried")
            goal = MoveBaseGoal()
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.header.frame_id = 'map'

            goal.target_pose.pose = self.pick_data.pose

            action_client.send_goal(goal)

            action_client.wait_for_result()

            self.state = action_client.get_state()
            if self.state ==3:
                print("state ==3")
                goToPick_tried = True
                self.done = True
                return pt.common.Status.SUCCESS

            self.tried = True
            

            return pt.common.Status.RUNNING

        # elif self.state == 3:
        #     print("state ==3")
        #     goToPick_tried = True
        #     self.done = True
        #     return pt.common.Status.SUCCESS

        elif self.state == 4: 
            print("state ==4")
            self.tried = False
            return pt.common.Status.FAILURE
        else:
            return pt.common.Status.RUNNING


class goToPlacePose(pt.behaviour.Behaviour):

    def __init__(self):

        rospy.loginfo("Initialising place cube behaviour.")
        

        # init the pick data to zeros
        self.place_data = PoseStamped()
        
        # execution checker
        self.tried = False
        self.done = False
        self.state = 0

        # become a behaviour
        super(goToPlacePose, self).__init__("goToPlacePose!")

    def getPlacePoseMessage(self,data):
        # print("this is the pick data")
        # print(data)
        self.place_data = data

    def update(self):
        global goToPlace_tried
        if goToPlace_tried == False:
            self.done = False
            self.tried = False

        # rospy.Subscriber('/place_pose',PoseStamped,callback=self.getPlacePoseMessage)
        # # # # init simple client for move base
        # # print("init simple client for move base")
        # action_client = SimpleActionClient("/move_base",MoveBaseAction)
        # action_client.wait_for_server(rospy.Duration(60))

        # goal = MoveBaseGoal()
        # goal.target_pose.header.stamp = rospy.Time.now()
        # goal.target_pose.header.frame_id = 'map'
        # # if self.pick_data.pose.orientation.x ==0 and self.pick_data.pose.orientation.y ==0 and self.pick_data.pose.orientation.z==0 and self.pick_data.pose.orientation.w==0:
        # #     goal.target_pose.pose.orientation.w = 1.0
        # # else:
        # goal.target_pose.pose = self.place_data.pose
        
        # # print("this is goal pose")
        # # print(self.pick_data.pose)
        # # print(goal.target_pose.pose)

        # action_client.send_goal(goal)

        # action_client.wait_for_result()

        # state = action_client.get_state()
        # print("action client state")
        # print(state)
        rospy.Subscriber('/place_pose',PoseStamped,callback=self.getPlacePoseMessage)
        # # # init simple client for move base
        # print("init simple client for move base")
        action_client = SimpleActionClient("/move_base",MoveBaseAction)
        action_client.wait_for_server()
        if self.done:
            print("go to place self done")
            return pt.common.Status.SUCCESS

        elif not goToPlace_tried:
            print("go to place not tried")
            goal = MoveBaseGoal()
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.header.frame_id = 'map'

            goal.target_pose.pose = self.place_data.pose
            
            # print("this is goal pose")
            # print(self.pick_data.pose)
            # print(goal.target_pose.pose)

            action_client.send_goal(goal)

            action_client.wait_for_result()

            self.state = action_client.get_state()
            if self.state == 3:
                print("state ==3")
                goToPlace_tried = True
                self.done = True      # clear map
                return pt.common.Status.SUCCESS
           

            self.tried = True
            return pt.common.Status.RUNNING

        # elif self.state == 3:
        #     print("state ==3")
        #     goToPlace_tried = True
        #     self.done = True
        #     return pt.common.Status.SUCCESS

        elif self.state == 4: 
            print("state ==4")
            self.tried = False
            return pt.common.Status.FAILURE
        else:
            return pt.common.Status.RUNNING
       


class BehaviourTree(ptr.trees.BehaviourTree):

	def __init__(self):

		rospy.loginfo("Initialising behaviour tree")

		rospy.Subscriber("/detected_aruco_pose",PoseStamped,callback=self.callback)
       	# tuck the arm
		# global_tried3 = False       
		# tuck the arm
		print("tuck arm")
		b1 = tuckarm1()


        # robot locolization
		b0 = robotLocalization()

        # move head
		print("move head up")
		b2 = move_headup1()


        # go to pick pose
		b3 = goToPickPose()

        # move head
		print("move head down")
		b4 = move_headdown1()

        # pick cube
		b5 = pick_cube()

        # move head
		b6 = move_headup2()

        # go to place pose
		b7 = goToPlacePose()

		# place cube
		b8 = place_cube()

        # move head
		print("move head down")
		b9 = move_headdown2()

        # detect cube on final table
		b10 = detect_cube_on_second_table()

        # # move head
		# print("move head down")
		# b11 = movehead1("up")	

        # # go to pick pose
		# b12 = goToPickPose()	

        # go to pick pose
		b13 = spawnGazeboModel()

		print("move backward")
		b12 = pt.composites.Selector(
			name="Go to door fallback", 
			children=[counter(70, "At second table?"), go("move backwards!", 0,1)]
		)
  
		# # become the tree
		# tree = RSequence(name="Main sequence", children=[b0])

		# become the tree
		tree = RSequence(name="Main sequence", children=[b0,b12,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b13])
        
       
		super(BehaviourTree, self).__init__(tree)
		# execute the behaviour tree

		rospy.sleep(5)
		self.setup(timeout=10000)
        
		# while not rospy.is_shutdown():
		#     print("this is b1 done")
		#     print(b1.name)
		#     self.tick_tock(1)
		#     print("this is b1 done")
		#     print(b1.done)
	
		while not rospy.is_shutdown():self.tick_tock(1)
        
		

	def callback(self,data):
		pub = rospy.Publisher('/marker_pose_topic',PoseStamped,queue_size=10)
		pub.publish(data)

        
	


if __name__ == "__main__":


	rospy.init_node('main_state_machine')
	try:
		BehaviourTree()
	except rospy.ROSInterruptException:
		pass

	rospy.spin()
