#!/usr/bin/python

import sys 
import numpy as np

def argParse():
    if len(sys.argv) < 2:
        print ("Usage: ./nnet.py N epochs eta ")
        exit(1)
    trainingX = np.loadtxt("TrainDigitX.csv.gz",delimiter=',')
    trainingY = np.loadtxt("TrainDigitY.csv.gz")
    return (trainingX,trainingY)

def loadTests():
    testingX = np.loadtxt("TestDigitX2.csv.gz",delimiter=',')
    return testingX

#######################
## Math Helper Funcs ##
#######################

def sig(alpha):
    return float(1)/(1+np.exp(-alpha))

def derivSig(alpha):
    return sig(alpha)*(1-sig(alpha))

def evaluate(inputVal,weights):
    for weight in weights:
        inputVal = sig(np.dot(weight,inputVal))
    return inputVal

########################
## Learning Functions ##
########################

def updateWeights(eta,delw,weights):
    newWeights = [w - eta*dw for w,dw in zip(weights,delw)]
    return newWeights

def backPropagation(data, label, eta, weights):
    labelVector = np.zeros([10])
    labelVector[label] = 1
    new_delw = [np.zeros(w.shape) for w in weights]
    activate = data 
    activateList = [data]
    intermediateList = []
    for w in weights:
        inter = np.dot(w,activateList[-1])
        activate = sig(inter)
        activateList.append(activate)
        intermediateList.append(inter)
    delt1 = np.multiply((activateList[2]-labelVector),derivSig(intermediateList[1]))
    new_delw[1] = np.outer(delt1, activateList[1].transpose())
    inter = intermediateList[-2]
    delt0 = np.multiply(np.dot(weights[-1].transpose(),delt1),(derivSig(intermediateList[-2])))
    new_delw[0] = np.outer(delt0, activateList[0].transpose())
    return new_delw

#######################
## Training Function ##
#######################

def stochasticGradientDescent(trainingData,epochs,eta,weights):
    for ep in range(epochs):
        for x,y in zip(trainingData[0],trainingData[1]):
            delw = backPropagation(x,y,eta,weights)
            weights = updateWeights(eta,delw,weights)
        print("epoch " + str(ep) + ": " + str(float(test(weights)/10000.0)))
    return weights

######################
## Testing Function ##
######################

def test(weights):
    testX = np.loadtxt("TestDigitX.csv.gz",delimiter=',')
    testY = np.loadtxt("TestDigitY.csv.gz")
    results = [(np.argmax(evaluate(x,weights)),int(y)) for x,y in zip(testX,testY)]
    summation = 0
    for (x,y) in results:
        if x==y:
            summation += 1
    return summation 

############################
## Network Specifications ##
############################
layers = 3
N = int(sys.argv[1]) 
layerSizes = [784,N,10]

weights = [np.random.randn(layerSizes[1],layerSizes[0]),np.random.randn(layerSizes[2],layerSizes[1])] 

############
##  Main  ##
############

epochs = int(sys.argv[2]) 
eta = float(sys.argv[3]) 
print ("N: " + str(N) + " Epochs: " + str(epochs) + " Eta: " + str(eta))

trainingData = argParse()
weights = stochasticGradientDescent(trainingData,epochs,eta,weights)
testingData = loadTests()
#for image in testingData:
#    prediction = np.argmax(evaluate(image,weights))
#    with open("ResultsX2.txt",'a+') as f:
#        f.write(str(prediction)+"\n")


