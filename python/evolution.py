# Make sure to have the server side running in V-REP: 
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

try:
	import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')
	
#import tensorflow as tf
import simulation
import random
#import create_model
import time

print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP

if clientID!=-1:
	#get initial model
	#first_model = create_model.getModel();
	first_model = [1,1,1,1,1,1,1,1,1,1,1,1]
	best_models = [first_model,first_model,first_model,first_model,first_model]
						
	for iteration in range(10):
		print ('Connected to remote API server')
		best_new_distance = []
		best_new_models = []
		for m, curr_model in enumerate(best_models):
			for permutation in range(len(curr_model)):
				model = curr_model
				if permutation != 1:
					model = curr_model #change to permuted model
					
				 # get real permuted model
				distanceTraveled = simulation.run_sim(model, clientID);
				
				for index in range(len(best_models)):
					if len(best_new_models) <= index:
						best_new_distance.append(distanceTraveled);
						best_new_models.append(distanceTraveled)
						break
					elif distanceTraveled > best_new_distance[index]:
						best_new_distance.insert(index, distanceTraveled);
						if len(best_new_distance) > len(best_models):
							best_new_distance.pop(len(best_models))
							
						best_new_models.insert(index, model)
						if len(best_new_models) > len(best_models):
							best_new_models.pop(len(best_models))
						break
						
		best_models = best_new_models
		print(best_new_distance)
	
    # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
	vrep.simxGetPingTime(clientID)

    # Now close the connection to V-REP:
	vrep.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')

