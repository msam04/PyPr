import csv
import pandas as pd
from sklearn.neural_net import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt


ouputObservations = []
##T1_values = []
##RH_1_values = []
##T2_values = []
##RH_2_values = []
##T3_values = []
##RH_3_values = []
##T4_values = []
##RH_4_values = []
##T5_values = []
##RH_5_values = []
##T6_values = []
##RH_6_values = []
##T7_values = []
##RH_7_values = []
##T8_values = []
##RH_8_values = []
##T9_values = []
##RH_9_values = []
##T_out_values = []
##Press_mm_hg_values = []
##RH_out_values = []
##Windspeed_values = []
##Visibility_values = []
##Tdewpoint_values = []
outputEnergyValues = []
##
with open('test.csv') as testDataFile:
    cvsReader = csv.reader(testDataFile, delimiter = ',', lineterminator = '\r\n')
    rowNum = 0
    for row in cvsReader:
       colNum = 0
       if rowNum == 0:
            colHeaders = row
       else:
            for col in row:
                if colNum == 0:             
                    outputObservations.append(col)
##                elif colNum == 1:
##                    T1_values.append(col)
##                elif colNum == 2:
##                    RH_1_values.append(col)
##                elif colNum == 3:
##                    T2_values.append(col)
##                elif colNum == 4:
##                    RH_2_values.append(col)
##                elif colNum == 5:
##                    T3_values.append(col)
##                elif colNum == 6:
##                    RH_3_values.append(col)
##                elif colNum == 7:
##                    T4_values.append(col)
##                elif colNum == 8:
##                    RH_4_values.append(col)
##                elif colNum == 9:
##                    T5_values.append(col)
##                elif colNum == 10:
##                    RH_5_values.append(col)
##                elif colNum == 11:
##                    T6_values.append(col)
##                elif colNum == 12:
##                    RH_6_values.append(col)
##                elif colNum == 13:
##                    T7_values.append(col)
##                elif colNum == 14:
##                    RH_7_values.append(col)
##                elif colNum == 15:
##                    T8_values.append(col)
##                elif colNum == 16:
##                    RH_8_values.append(col)
##                elif colNum == 17:
##                    T9_values.append(col)
##                elif colNum == 18:
##                    RH_9_values.append(col)
##                elif colNum == 19:
##                    T_out_values.append(col)
##                elif colNum == 20:
##                    Press_mm_hg_values.append(col)
##                elif colNum == 21:
##                    RH_out_values.append(col)
##                elif colNum == 22:
##                    Windspeed_values.append(col)
##                elif colNum == 23:
##                    Visibility_values.append(col)
##                elif colNum == 24:
##                    Tdewpoint_values.append(col)
##                elif colNum == 25:
##                    Energy_values.append(col)
                colNum += 1
##            #print(rowNum)
    rowNum += 1

    testDataFile.close()

with open('submission.csv','w') as outputFile:
    fieldNames = ['Observation', 'Energy']
    writer = csv.DictWriter(outputFile, filednames = fieldNames)
    writer.writeheader()

    outputFile.close()
##
####print(colHeaders)
####print(Observations[0])
####print(T1_values[0])
####print(RH_1_values[0])
####print(T2_values[0])
####print(RH_2_values[0])
####print(T3_values[0])
####print(RH_3_values[0])
####print(T4_values[0])
####print(RH_4_values[0])
####print(T5_values[0])
####print(RH_5_values[0])
####print(T6_values[0])
####print(RH_6_values[0])
####print(T7_values[0])
####print(RH_7_values[0])
####print(T8_values[0])
####print(RH_8_values[0])
####print(T9_values[0])
####print(RH_9_values[0])
####print(T_out_values[0])
####print(Press_mm_hg_values[0])
####print(RH_out_values[0])
####print(Windspeed_values[0])
####print(Visibility_values[0])
####print(Tdewpoint_values[0])
####print(Energy_values[0])
##
###for val in Observations:
##    #print(val)
##    
    
        
trainData = pd.read_csv('train.csv', names = ["Observation","T1","RH_1","T2","RH_2","T3","RH_3","T4","RH_4","T5","RH_5","T6","RH_6","T7","RH_7","T8","RH_8","T9","RH_9","T_out","Press_mm_hg","RH_out","Windspeed","Visibility","Tdewpoint","Energy"])
trainData.head()
xTrain = trainData.drop("Energy", axis = 1)
yTrain = trainData["Energy"]
testData = pd.read_csv('train.csv', names = ["Observation","T1","RH_1","T2","RH_2","T3","RH_3","T4","RH_4","T5","RH_5","T6","RH_6","T7","RH_7","T8","RH_8","T9","RH_9","T_out","Press_mm_hg","RH_out","Windspeed","Visibility","Tdewpoint","Energy"])
testData.head()
xTest = testData.drop("Energy", axis = 1)
yTest = testData["Energy"]
neuralNet = MLPRegressor(hidden_layer_sizes = (15), activation = 'logistic', solver = 'sgd', learning_rate = 'adaptive')
n = neuralNet.fit(xTrain, yTrain)
testResult = neuralNet.predict(testX)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=5, c='b', marker="o", label='real')
ax1.plot(test_x,test_y, c='r', label='NN Prediction')


    

plt.legend()
plt.show()

with open('submission.csv','w') as outputFile:
    fieldNames = ['Observation', 'Energy']
    writer = csv.DictWriter(outputFile, filednames = fieldNames)
    writer.writeheader()
    outputRowNum = 1
    

    outputFile.close()

