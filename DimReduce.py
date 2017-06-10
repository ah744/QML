#!/usr/bin/python
import numpy as np
import sys 
from matplotlib import pyplot as plt

N = 500 

######### Helper and PCA Functions ###############

def loadData():
    data = []
    colors = []
    with open("3Ddata.txt") as f:
        for line in f:
            lineSplit = line.split()[:-1]
            data.append([float(val) for val in lineSplit])
            colors.append(int(line.split()[-1]))
    data = np.array(data).transpose()
    return data, colors

def centerData(data):
    mean = np.array([np.mean(data[0,:]),np.mean(data[1,:]),np.mean(data[2,:])])
    centeredData = data[:]
    centeredData[0,:] = data[0,:] - mean[0]
    centeredData[1,:] = data[1,:] - mean[1]
    centeredData[2,:] = data[2,:] - mean[2]
    return centeredData 

def computeSampleCovarianceMatrix(data):
    x = data[0,:]
    y = data[1,:]
    z = data[2,:]
    return np.cov([x,y,z])

def findAndSortEigenvalues(matrix):
    eigenvalues,eigenvectors = np.linalg.eig(matrix)
    eigensets = [(np.abs(eigenvalues[i]),eigenvectors[:,i]) for i in range(eigenvectors.shape[1])]
    eigensets.sort(key=lambda x:x[0], reverse=True)
    return eigensets 

def makePlot(data,colors,name):
    colorChoices = ['green', 'yellow', 'blue', 'red']
    for i in range(data.shape[1]):
        plt.plot(data[0,i], data[1,i], 'o', markersize=7, color=colorChoices[colors[i]-1], alpha=0.5, label='data')
    plt.xlabel('x coordinates')
    plt.ylabel('y coordinates')
    plt.title(name + ' Dimensionality Reduction')
    plt.show()

def makePlotPerceptron(data):
    plt.plot(data[1,:], data[0,:], 'o', markersize=7, color='blue', alpha=0.5, label='data')
    plt.xlabel('examples seen')
    plt.ylabel('errors')
    plt.title("Perceptron Errors By Examples Seen")
    plt.show()

def PCA():
    data,colors = loadData()
    centeredData = centerData(data)
    cov = computeSampleCovarianceMatrix(data)
    eigensets = findAndSortEigenvalues(cov)
    pcomponents = []
    pcomponents.append(eigensets[0][1])
    pcomponents.append(eigensets[1][1])
    pcomponents = np.array(pcomponents).transpose()
    embedding = (centeredData.transpose().dot(pcomponents)).transpose()
    makePlot(embedding,colors,'PCA')

######### Isomap Functions ###############

def MDS(distance):
    P = np.identity(distance.shape[1]) - np.ones((distance.shape[1],distance.shape[1]))/distance.shape[1] 
    G = -1./2 * P.dot(distance*distance).dot(P)
    eigsystem = findAndSortEigenvalues(G)
    Y = [(eigsystem[0][0]**0.5 * eigsystem[0][1][i],eigsystem[1][0]**0.5 * eigsystem[1][1][i]) for i in range(distance.shape[1])]
    Y = np.array(Y).transpose()
    print Y
    return Y

def FloydWarshall(graph):
    for k in range(graph.shape[1]):
        for i in range(graph.shape[1]):
            for j in range(graph.shape[1]):
                if graph[i][j] > graph[i][k] + graph[k][j]:
                    graph[i][j] = graph[i][k] + graph[k][j]
    return graph

def buildGraph(data):
    graph = np.full((data.shape[1],data.shape[1]),sys.maxint)
    for i in range(data.shape[1]):
        distances = []
        for j in range(data.shape[1]):
            distances.append((np.linalg.norm(data[:,i]-data[:,j]),j))
        distances.sort(key=lambda x:x[0])
        nearest = [x for x in distances[1:11]]
        for pair in nearest:
            graph[i][pair[1]] = pair[0] 
        graph[i,i] = 0
    return graph

def Isomap():
    data,colors = loadData()
    centeredData = centerData(data)
    graph = buildGraph(centeredData)
    distance = FloydWarshall(graph)
    embedding = MDS(distance)
    makePlot(embedding,colors, "Isomap")

######### Locally Linear Embedding Functions ###############

def calculateWeights(data, graph):
    W = np.full((N,N),0)
    for i in range(N): 
        neighbors = [j for j in range(N) if graph[i][j] < sys.maxint and graph[i][j] > 0]
        nMatrix = [data[:,k] for k in neighbors]
        nMatrix = np.array(nMatrix).transpose()
        nMatrix = np.subtract(nMatrix.transpose(),data[:,i]).transpose()
        cov = np.cov(nMatrix) 
        pad = np.full((10,10),0)
        pad[:cov.shape[0],:cov.shape[1]] = cov
        e = np.trace(cov)*(10**-3)
        pad += e*np.identity(10)
        w = np.linalg.solve(pad,np.ones(10))
        k = 0
        for j in neighbors:
            W[i,j] = w[k]/np.sum(w)
            k += 1
    print "Weights"
    print W
    return W

def calculateEmbedding(W):
    G = (np.identity(N) - W).transpose().dot(np.identity(N)-W)
    eigensystem = findAndSortEigenvalues(G)
    Y = []
    Y.append(eigensystem[-2][1])
    Y.append(eigensystem[-3][1])
    Y = np.array(Y)
    print "Eigensystem:"
    print eigensystem[-2][0]
    print eigensystem[-2][1]
    print eigensystem[-3][0]
    print eigensystem[-3][1]
    return Y

def LLE():
    data,colors = loadData()
    centeredData = centerData(data)
    graph = buildGraph(centeredData)
    W = calculateWeights(centeredData, graph)
    Y = calculateEmbedding(W)
    makePlot(Y,colors, "LLE")

######### Perceptron Functions ###############

def loadDigits(name):
    data = []
    with open(name) as f:
        for line in f:
            data.append([int(val) for val in line.split()])
    return data

def loadLabels():
    labels = []
    with open("train35.labels") as f:
        for line in f:
            labels.append(int(line))
    return labels 

def makePrediction(weights, vector):
    if weights.dot(vector) >= 0:
        return 1
    else:
        return -1 

def updateWeights(prediction, answer, vector, weights):
    error = 0
    if prediction < answer:
        error = 1
        weights += vector 
    elif prediction > answer:
        error = 1
        weights -= vector 
    return error, weights

def crossValidate(data):
    labels = loadLabels()
    lowError = sys.maxint 
    lowM = 0
    for M in range(1,10):
        errorList = []
        for partition in range(4):
            lowIdx = partition*400
            highIdx = (partition+1)*400
            removedPartition = data[0:lowIdx]
            removedPartition.extend(data[highIdx:])
            errors, labels = Perceptron(removedPartition, labels, M, False)
            errorList.append(float(errors[-1][0])/errors[-1][1])
        errorList = np.array(errorList)
        average = np.mean(errorList)
        if average < lowError:
            lowError = average
            lowM = M
    return lowM

def Perceptron(data, labels, M, final):
    errors = [(0,0)]
    weights = np.zeros(784) 
    answerLabels = []
    sample = 0
    for i in range(M):
        for j in range(len(data)):
            sample += 1
            prediction = makePrediction(weights, np.array(data[j]))
            error,weights = updateWeights(prediction, labels[j], np.array(data[j]), weights)
            errors.append((errors[-1][0]+error,sample))
    if final:
        test = loadDigits("test35.digits")
        answerLabels = []
        for i in range(len(test)):
            answerLabels.append(makePrediction(weights, np.array(test[i])))
        errors = np.array(errors).transpose()
        makePlotPerceptron(errors)
        with open("test35.predictions", 'w') as out:
            for val in answerLabels:
                out.write(str(val) + "\n")
    return (errors,labels)

################ Main #######################

PCA()
Isomap()
LLE()
#### Perceptron ####
data = loadDigits("train35.digits")
labels = loadLabels()
test = loadDigits("test35.digits")
M = crossValidate(data)
final = True
Perceptron(data, labels, M, final)

