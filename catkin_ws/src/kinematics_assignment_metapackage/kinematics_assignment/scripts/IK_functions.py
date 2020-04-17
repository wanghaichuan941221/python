#! /usr/bin/env python3

"""
    # {Haichuan Wang}
    # {haichuan@kth.se}
"""

import numpy as np
import math
from math import cos,sin

old_p = np.zeros(6)

def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]
    q = [0.0, 0.0, 0.0]

    l0 = 0.07
    l1 = 0.3
    l2 = 0.35
    x = x - l0

    # the radius of l1,l2
    r = np.sqrt(x*x+y*y)
    #law of cosine
    cos_theta2 = (r*r-l1*l1-l2*l2)/(2*l1*l2)
     
    q[1] = np.arctan2(np.sqrt(1-cos_theta2**2),cos_theta2)

    q[0] = np.arctan2(y,x)-np.arctan2(l2*np.sin(q[1]),l1+l2*np.cos(q[1]))

    q[2] = z
    """
    Fill in your IK solution here and return the three joint values in q
    """

    return q


#calculate the forward kinematics of robotiics
def forward_kinematics(q,d,alpha,a=0):
    #define jocobian matrix
    pz = np.zeros((3,7))
    zs = np.zeros((7,3))
    #Re is rotation matrix of end effector
    Re = np.zeros((3,3))
    #T is the translation matrix from joint i to joint i+1
    T = np.zeros((7,4,4))
    for i in range(7):
        T[i,:,:] = np.array([
            [cos(q[i]),-sin(q[i])*cos(alpha[i]),sin(q[i])*sin(alpha[i]),0],
            [sin(q[i]),cos(q[i])*cos(alpha[i]),-cos(q[i])*sin(alpha[i]),0],
            [0,sin(alpha[i]),cos(alpha[i]),d[i]],
            [0,0,0,1]
        ])
    
    #T1 is the transform matrix of all joints
    T1 = np.zeros((7,4,4))
    T1[0,:,:]=T[0,:,:]
    for i in range(1,7):
        T1[i,:,:]=np.dot(T1[i-1,:,:],T[i,:,:])

    #get ratation matrix
    Re = T1[6,0:3,0:3]
    #get the joint translation pz and rotation of z-axis zs  
    for i in range(7):
        pz[:,i] = T1[i,0:3,3]
        zs[i,:] = T1[i,0:3,2]

    return pz,zs,Re


#calculate angle according to book ch3.7
def angle_error(R,Re):
    nd,sd,ad = np.array([R[0][0],R[1][0],R[2][0]]),np.array([R[0][1],R[1][1],R[2][1]]),np.array([R[0][2],R[1][2],R[2][2]])
    ne,se,ae = Re[:,0],Re[:,1],Re[:,2]
    e = 0.5*(np.cross(ne,nd)+np.cross(se,sd)+np.cross(ae,ad))
    return e

def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    # q = [0,0,0,0,0,0,0] # set q to zeros for debug purpose
    q=joint_positions #it must contain 7 elements

    """
    Fill in your IK solution here and return the seven joint values in q
    """
    l0 = 0.311
    L = 0.4
    M = 0.39
    l6 = 0.078
    pi_over_2 = math.pi/2

    #d is the given matrix of arm length
    d = np.array([l0,0,L,0,M,0,l6])
    #alpha is the joint angle
    alpha = np.array([pi_over_2,-pi_over_2,-pi_over_2,pi_over_2,pi_over_2,-pi_over_2,0])

    while True:
        #get the translation position and rotation of z-axis of forward kinematics
        #Re is the rotation matrix of end effector
        #pz is the translation position
        # zs is the rotation matrix of z-axis 
        pz,zs,Re = forward_kinematics(q,d,alpha)
        # pe is the end effector
        pe = pz[:,6]
        # calculate angle error according to book ch3.7
        err_a = angle_error(R,Re)

        J = np.zeros((6,7))
        p0=np.array([0,0,0])
        z0 = np.array([0,0,1])
        #calculate the J0 using initial p0 and z0
        J[0:3,0]=np.cross(z0,pe-p0)
        J[3:6,0]=z0

        #calculate Jocobian according book ch3.1
        for i in range(1,7):
            J[0:3,i] = np.cross(zs[i-1],pe-pz[:,i-1],axis=0)
            J[3:6,i] = zs[i-1]
        #calculate the pseudo inverse
        inv_J = np.linalg.pinv(J)

        # calculate the translation error
        p_goal = np.array([x,y,z])
        p_current = np.array([pe[0],pe[1],pe[2]])
        delta_d = p_current - p_goal

        #merge the translation error with angle error
        delta_x = np.zeros(6)
        delta_x[0:3]=delta_d
        delta_x[3:6]=err_a

        #calculate the position error between desired position and current position 
        q -= np.dot(inv_J,delta_x.reshape(6,1)).T[0]

        #if the error is smaller than the tolerance error, break the while loop
        if np.linalg.norm(delta_x)<0.01:
            # print("test")
            break

        # #some debug code 
        # # print the jocobian matrix --->set q to zeros to check whether jocobian matrix is correct
        # print("Jocobian")
        # print(J)
        # #print the delta_x for q is zero
        # print("position error")
        # print(delta_x)
        # #some other debug code ----> to be dome 
        # #exit the whole function
        # exit()

    return q
