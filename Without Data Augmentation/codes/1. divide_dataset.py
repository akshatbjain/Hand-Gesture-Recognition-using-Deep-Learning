# This code divides the entire dataset into three parts: Training (60%), Validation (20%) and Testing (20%)

from shutil import copyfile
import random
import os

# Location where the entire dataset is present
Gest = ['/home/akshat/deep_learning/Consolidated_Data_Set/Gesture_1','/home/akshat/deep_learning/Consolidated_Data_Set/Gesture_2','/home/akshat/deep_learning/Consolidated_Data_Set/Gesture_3','/home/akshat/deep_learning/Consolidated_Data_Set/Gesture_4','/home/akshat/deep_learning/Consolidated_Data_Set/Gesture_5',]

# Locations where the divided dataset is being saved
Test = ['/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/test/Gesture_1','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/test/Gesture_2','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/test/Gesture_3','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/test/Gesture_4','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/test/Gesture_5']

Validation = ['/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/validation/Gesture_1','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/validation/Gesture_2','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/validation/Gesture_3','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/validation/Gesture_4','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/validation/Gesture_5']

Training = ['/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/training/Gesture_1','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/training/Gesture_2','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/training/Gesture_3','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/training/Gesture_4','/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/training/Gesture_5']

FileList = [0]*5
for i in xrange(5):
    FileList[i] = os.listdir(Gest[i])
    random.shuffle(FileList[i])


for i in xrange(len(FileList)):
    leng = int(len(FileList[i]))
    for j in xrange(leng):
	if(j<leng*0.6):	
            copyfile(Gest[i]+'/'+FileList[i][j], Training[i]+'/'+FileList[i][j])
	elif(j<0.8*leng):
	    copyfile(Gest[i]+'/'+FileList[i][j], Test[i]+'/'+FileList[i][j])
	else:
	    copyfile(Gest[i]+'/'+FileList[i][j], Validation[i]+'/'+FileList[i][j])
