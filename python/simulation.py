try:
	import vrep
	import time
	import tensorflow as tf
	import pandas as pd
	import model_iterator
	import math
	import operator
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

def run_sim(model, clientID):
	vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait) # start simulation
	
	previousPosition = [-2.71879,-1.90075,36.263,1.35473,0.780061,-35.7055,-0.225083,-3.02226,1.82057,0.258116,-2.12325,1.061]
	
	#RhipX,RhipY,RhipZ,LhipX,LhipY,LhipZ,RKneeZ,LKneeZ,RAnkleX,RAnkleZ,LAnkleX,LAnkleZ
	legJoints = [0,0,0,0,0,0,0,0,0,0,0,0]
	legJointInv = [1,1,1,-1,-1,1,1,1,1,1,-1,1]
	hipJointInv = [1,1,1,1,1,-1,1,1,1,1,1,1]
	
	returnCode01, legJoints[0] = vrep.simxGetObjectHandle(clientID, "rightLegJoint0", vrep.simx_opmode_blocking)
	returnCode02, legJoints[1] = vrep.simxGetObjectHandle(clientID, "rightLegJoint1", vrep.simx_opmode_blocking)
	returnCode03, legJoints[2] = vrep.simxGetObjectHandle(clientID, "rightLegJoint2", vrep.simx_opmode_blocking)
	returnCode04, legJoints[6] = vrep.simxGetObjectHandle(clientID, "rightLegJoint3", vrep.simx_opmode_blocking) 
	returnCode05, legJoints[8] = vrep.simxGetObjectHandle(clientID, "rightLegJoint5", vrep.simx_opmode_blocking)
	returnCode06, legJoints[9] = vrep.simxGetObjectHandle(clientID, "rightLegJoint4", vrep.simx_opmode_blocking)
	
	returnCode07, legJoints[3] = vrep.simxGetObjectHandle(clientID, "leftLegJoint0", vrep.simx_opmode_blocking)
	returnCode08, legJoints[4] = vrep.simxGetObjectHandle(clientID, "leftLegJoint1", vrep.simx_opmode_blocking)
	returnCode09, legJoints[5] = vrep.simxGetObjectHandle(clientID, "leftLegJoint2", vrep.simx_opmode_blocking)
	returnCode010, legJoints[7] = vrep.simxGetObjectHandle(clientID, "leftLegJoint3", vrep.simx_opmode_blocking) 
	returnCode011, legJoints[10] = vrep.simxGetObjectHandle(clientID, "leftLegJoint5", vrep.simx_opmode_blocking)
	returnCode012, legJoints[11] = vrep.simxGetObjectHandle(clientID, "leftLegJoint4", vrep.simx_opmode_blocking)
	
	returnCode013, head = vrep.simxGetObjectHandle(clientID, "Asti", vrep.simx_opmode_blocking)
	returnCode013, r_foot = vrep.simxGetObjectHandle(clientID, "rightFoot", vrep.simx_opmode_blocking)
	returnCode013, l_foot = vrep.simxGetObjectHandle(clientID, "leftFoot", vrep.simx_opmode_blocking)
	
	#RhipX,RhipY,RhipZ,LhipX,LhipY,LhipZ,RKneeZ,LKneeZ,RAnkleX,RAnkleZ,LAnkleX,LAnkleZ
	legPositions = [0,0,0,0,0,0,0,0,0,0,0,0]
	
	headLocation = [1,1,1];
	begin_r = [0,0,0]
	begin_l = [0,0,0]
	end_r = [0,0,0]
	end_r = [0,0,0]
	
	for i in range(12):
		returnCode, legPositions[i] = vrep.simxGetJointPosition(clientID, legJoints[i], vrep.simx_opmode_streaming)
		returncode, headLocation = vrep.simxGetObjectPosition(clientID, head, -1, vrep.simx_opmode_streaming)
		returnCode, begin_r =  vrep.simxGetObjectPosition(clientID, r_foot, -1, vrep.simx_opmode_streaming)
		returnCode, begin_l =  vrep.simxGetObjectPosition(clientID, l_foot, -1, vrep.simx_opmode_streaming)
	
	start_time = int(round(time.time()))
	while vrep.simxGetConnectionId(clientID) != -1 and ((headLocation[2] > 0.0 or headLocation[2] == 0.0) and ((int(round(time.time())) - start_time) < 100)):
		for i in range(12):
			returnCode, legPositions[i] = vrep.simxGetJointPosition(clientID, legJoints[i], vrep.simx_opmode_buffer)
		
		# do something to the joints dummy right now
		#remember to convert to radians and back
		degreeLegPositions = [0,0,0,0,0,0,0,0,0,0,0,0]
		for i in range(len(legPositions)):
			degreeLegPositions[i] = legPositions[i] * legJointInv[i] * 180 / math.pi
		
		inputData = []
		for element in degreeLegPositions:
			inputData.append(element)
		
		newJointPositions = model_iterator.runModel(model,[tuple(inputData)])
		
		newJointPositions = newJointPositions[0]
		
		for i in range(len(newJointPositions)):
			newJointPositions[i] = newJointPositions[i] * math.pi / 180
		
		vrep.simxPauseCommunication(clientID,1)
		for i in range(12):
			vrep.simxSetJointTargetPosition(clientID,legJoints[i],newJointPositions[i] * legJointInv[i] * hipJointInv[i],vrep.simx_opmode_oneshot)
		vrep.simxPauseCommunication(clientID,0)
		time.sleep(0.001)
		
		returncode, headLocation = vrep.simxGetObjectPosition(clientID, head, -1, vrep.simx_opmode_buffer)
	
	returnCode, end_r =  vrep.simxGetObjectPosition(clientID, r_foot, -1, vrep.simx_opmode_buffer)
	returnCode, end_l =  vrep.simxGetObjectPosition(clientID, l_foot, -1, vrep.simx_opmode_buffer)
	
	distanceTraveled = (((end_r[1] + end_l[1])/2) - ((begin_r[1] + begin_l[1])/2))
			
	vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
	time.sleep(0.25)
	
	return distanceTraveled
	
def run_sim_direction(model, clientID):
	vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait) # start simulation
	
	previousPosition = [-2.71879,-1.90075,36.263,1.35473,0.780061,-35.7055,-0.225083,-3.02226,1.82057,0.258116,-2.12325,1.061]
	
	#RhipX,RhipY,RhipZ,LhipX,LhipY,LhipZ,RKneeZ,LKneeZ,RAnkleX,RAnkleZ,LAnkleX,LAnkleZ
	legJoints = [0,0,0,0,0,0,0,0,0,0,0,0]
	legJointInv = [1,1,1,-1,-1,1,1,1,1,1,-1,1]
	
	returnCode01, legJoints[0] = vrep.simxGetObjectHandle(clientID, "rightLegJoint0", vrep.simx_opmode_blocking)
	returnCode02, legJoints[1] = vrep.simxGetObjectHandle(clientID, "rightLegJoint1", vrep.simx_opmode_blocking)
	returnCode03, legJoints[2] = vrep.simxGetObjectHandle(clientID, "rightLegJoint2", vrep.simx_opmode_blocking)
	returnCode04, legJoints[6] = vrep.simxGetObjectHandle(clientID, "rightLegJoint3", vrep.simx_opmode_blocking) 
	returnCode05, legJoints[8] = vrep.simxGetObjectHandle(clientID, "rightLegJoint5", vrep.simx_opmode_blocking)
	returnCode06, legJoints[9] = vrep.simxGetObjectHandle(clientID, "rightLegJoint4", vrep.simx_opmode_blocking)
	
	returnCode07, legJoints[3] = vrep.simxGetObjectHandle(clientID, "leftLegJoint0", vrep.simx_opmode_blocking)
	returnCode08, legJoints[4] = vrep.simxGetObjectHandle(clientID, "leftLegJoint1", vrep.simx_opmode_blocking)
	returnCode09, legJoints[5] = vrep.simxGetObjectHandle(clientID, "leftLegJoint2", vrep.simx_opmode_blocking)
	returnCode010, legJoints[7] = vrep.simxGetObjectHandle(clientID, "leftLegJoint3", vrep.simx_opmode_blocking) 
	returnCode011, legJoints[10] = vrep.simxGetObjectHandle(clientID, "leftLegJoint5", vrep.simx_opmode_blocking)
	returnCode012, legJoints[11] = vrep.simxGetObjectHandle(clientID, "leftLegJoint4", vrep.simx_opmode_blocking)
	
	returnCode013, head = vrep.simxGetObjectHandle(clientID, "Asti", vrep.simx_opmode_blocking)
	returnCode013, r_foot = vrep.simxGetObjectHandle(clientID, "rightFoot", vrep.simx_opmode_blocking)
	returnCode013, l_foot = vrep.simxGetObjectHandle(clientID, "leftFoot", vrep.simx_opmode_blocking)
	
	#RhipX,RhipY,RhipZ,LhipX,LhipY,LhipZ,RKneeZ,LKneeZ,RAnkleX,RAnkleZ,LAnkleX,LAnkleZ
	legPositions = [0,0,0,0,0,0,0,0,0,0,0,0]
	
	headLocation = [1,1,1];
	begin_r = [0,0,0]
	begin_l = [0,0,0]
	end_r = [0,0,0]
	end_r = [0,0,0]
	
	for i in range(12):
		returnCode, legPositions[i] = vrep.simxGetJointPosition(clientID, legJoints[i], vrep.simx_opmode_streaming)
		returncode, headLocation = vrep.simxGetObjectPosition(clientID, head, -1, vrep.simx_opmode_streaming)
		returnCode, begin_r =  vrep.simxGetObjectPosition(clientID, r_foot, -1, vrep.simx_opmode_streaming)
		returnCode, begin_l =  vrep.simxGetObjectPosition(clientID, l_foot, -1, vrep.simx_opmode_streaming)
	
	start_time = int(round(time.time()))
	while vrep.simxGetConnectionId(clientID) != -1 and ((headLocation[2] > 0.5 or headLocation[2] == 0.0) and ((int(round(time.time())) - start_time) < 5)):
		for i in range(12):
			returnCode, legPositions[i] = vrep.simxGetJointPosition(clientID, legJoints[i], vrep.simx_opmode_buffer)
		
		# do something to the joints dummy right now
		#remember to convert to radians and back
		degreeLegPositions = [0,0,0,0,0,0,0,0,0,0,0,0]
		for i in range(len(legPositions)):
			degreeLegPositions[i] = legPositions[i] * legJointInv[i] * 180 / math.pi
		
		inputData = []
		for element in degreeLegPositions:
			inputData.append(element)
			
		for element in previousPosition:
			inputData.append(element)
		
		previousPosition = degreeLegPositions
		
		newJointPositions = model_iterator.runModel(model,[tuple(inputData)])
		
		newJointPositions = newJointPositions[0]
		
		for i in range(len(newJointPositions)):
			newJointPositions[i] = newJointPositions[i] * math.pi / 180
		
		vrep.simxPauseCommunication(clientID,1)
		for i in range(12):
			vrep.simxSetJointTargetPosition(clientID,legJoints[i],newJointPositions[i] * legJointInv[i],vrep.simx_opmode_oneshot)
		vrep.simxPauseCommunication(clientID,0)
		time.sleep(0.01)
		
		returncode, headLocation = vrep.simxGetObjectPosition(clientID, head, -1, vrep.simx_opmode_buffer)
	
	returnCode, end_r =  vrep.simxGetObjectPosition(clientID, r_foot, -1, vrep.simx_opmode_buffer)
	returnCode, end_l =  vrep.simxGetObjectPosition(clientID, l_foot, -1, vrep.simx_opmode_buffer)
	
	distanceTraveled = (((end_r[1] + end_l[1])/2) - ((begin_r[1] + begin_l[1])/2))
			
	vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
	time.sleep(0.25)
	
	return distanceTraveled
	
def run_sim_data(subject, clientID):
	vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait) # start 
		
	#RhipX,RhipY,RhipZ,LhipX,LhipY,LhipZ,RKneeZ,LKneeZ,RAnkleX,RAnkleZ,LAnkleX,LAnkleZ
	legJoints = [0,0,0,0,0,0,0,0,0,0,0,0]
	#RhipX,RhipY,RhipZ,LhipX,LhipY,LhipZ,RKneeZ,LKneeZ,RAnkleX,RAnkleZ,LAnkleX,LAnkleZ
	legPositions = [0,0,0,0,0,0,0,0,0,0,0,0]
	legJointInv = [1,1,1,-1,-1,1,1,1,1,1,-1,1]
	hipJointInv = [1,1,1,1,1,1,1,1,1,1,1,1]
	
	returnCode01, legJoints[0] = vrep.simxGetObjectHandle(clientID, "rightLegJoint0", vrep.simx_opmode_blocking)
	returnCode02, legJoints[1] = vrep.simxGetObjectHandle(clientID, "rightLegJoint1", vrep.simx_opmode_blocking)
	returnCode03, legJoints[2] = vrep.simxGetObjectHandle(clientID, "rightLegJoint2", vrep.simx_opmode_blocking)
	returnCode04, legJoints[6] = vrep.simxGetObjectHandle(clientID, "rightLegJoint3", vrep.simx_opmode_blocking) 
	returnCode05, legJoints[8] = vrep.simxGetObjectHandle(clientID, "rightLegJoint5", vrep.simx_opmode_blocking)
	returnCode06, legJoints[9] = vrep.simxGetObjectHandle(clientID, "rightLegJoint4", vrep.simx_opmode_blocking)
	
	returnCode07, legJoints[3] = vrep.simxGetObjectHandle(clientID, "leftLegJoint0", vrep.simx_opmode_blocking)
	returnCode08, legJoints[4] = vrep.simxGetObjectHandle(clientID, "leftLegJoint1", vrep.simx_opmode_blocking)
	returnCode09, legJoints[5] = vrep.simxGetObjectHandle(clientID, "leftLegJoint2", vrep.simx_opmode_blocking)
	returnCode010, legJoints[7] = vrep.simxGetObjectHandle(clientID, "leftLegJoint3", vrep.simx_opmode_blocking) 
	returnCode011, legJoints[10] = vrep.simxGetObjectHandle(clientID, "leftLegJoint5", vrep.simx_opmode_blocking)
	returnCode012, legJoints[11] = vrep.simxGetObjectHandle(clientID, "leftLegJoint4", vrep.simx_opmode_blocking)
	
	returnCode013, head = vrep.simxGetObjectHandle(clientID, "Asti", vrep.simx_opmode_blocking)
	
	headLocation = [1,1,1];
	returncode, headLocation = vrep.simxGetObjectPosition(clientID, head, -1, vrep.simx_opmode_streaming)
	
	for i in range(12):
		returnCode, legPositions[i] = vrep.simxGetJointPosition(clientID, legJoints[i], vrep.simx_opmode_streaming)
	
	#RhipX,RhipY,RhipZ,LhipX,LhipY,LhipZ,RKneeZ,LKneeZ,RAnkleX,RAnkleZ,LAnkleX,LAnkleZ
	start_time = int(round(time.time()))
	while vrep.simxGetConnectionId(clientID) != -1 and ((headLocation[2] > 0.3 or headLocation[2] == 0.0) and ((int(round(time.time())) - start_time) < 100)):
		for j in range(((subject * 101)),((subject * 101) + 202)):
			
			for i in range(12):
				returnCode, legPositions[i] = vrep.simxGetJointPosition(clientID, legJoints[i], vrep.simx_opmode_buffer)
			
			joint_data = model_iterator.getActual(j)
			newJointPositions = [0,0,0,0,0,0,0,0,0,0,0,0]
			
			for i in range(len(newJointPositions)):
				joint_data[i] = joint_data[i] * math.pi / 180
				newJointPositions[i] = joint_data[i] * legJointInv[i] * hipJointInv[i]
			print(j)
			print(newJointPositions)
			print()
			
			vrep.simxPauseCommunication(clientID,1)
			for i in range(12):
				vrep.simxSetJointTargetPosition(clientID,legJoints[i],newJointPositions[i],vrep.simx_opmode_oneshot)
			vrep.simxPauseCommunication(clientID,0)
			time.sleep(0.01)
			
	vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
	time.sleep(0.25)
	
	return 