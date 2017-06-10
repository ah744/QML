#!/usr/bin/python

import csv
import sys
import random
import numpy as np

K = 3 
def prepInputsKMeans():
    data = []
    clusters = {} 
    clusterSets = {}
    centers = []
    with open("toydata.txt") as f:
        for line in f:
            lineSplit = line.split()
            data.append((float(lineSplit[0]), float(lineSplit[1])))
            clusters[(float(lineSplit[0]),float(lineSplit[1]))] = -1
    for i in range(K):
        centers.append((random.uniform(-4,8),random.uniform(-4,8)))
    return (data,clusters,clusterSets,centers)

def dist((x1,y1),(x2,y2)):
    return (float(x2)-float(x1))**2 + (float(y2)-float(y1))**2

def findClosestCenter(element,centers):
    minDist = sys.maxint 
    minCenter = -1
    for i in range(len(centers)): 
        if(dist(element,centers[i])) <= minDist:
            minCenter = i
            minDist = dist(element,centers[minCenter])
    return minCenter 

def converged(newClusters,oldClusters):
    convergence = True
    return np.array_equal(newClusters,oldClusters) 

def prepareClusters(clusters, data):
    clusters = []
    for element in data:
        clusters.append([-1])
    return clusters

def updateCenters(centers, clusterSets):
    for j in range(K):
        if ((j not in clusterSets) or (len(clusterSets[j]) == 0)):
            continue
        else:
            centers[j] = (np.sum(clusterSets[j], axis=0))/(len(clusterSets[j]))
    return centers

def calculateDistortion(clusters,centers):
    totalDistortion = 0
    for element in clusters.values():
        for point in element:
            minDist = sys.maxint
            for centroid in centers:
                if (dist(point,centroid)) <= minDist:
                    minDist = dist(point,centroid)
            totalDistortion += minDist
    return totalDistortion

def kmeans(data,clusters,clusterSets,centers):
    oldCenters = []
    distortion = []
    for i in range(K):
        oldCenters.append((-1,-1))
    while(not converged(centers, oldCenters)):
        oldCenters = centers[:]
        clusterSets = {}
        for element in data:
           index = findClosestCenter(element,centers)
           clusters[element] = index
           try: 
               clusterSets[index].append((float(element[0]),float(element[1])))
           except KeyError:
               clusterSets[index] = [(float(element[0]),float(element[1]))]
        centers = updateCenters(centers,clusterSets)
        distortion.append(calculateDistortion(clusterSets,centers))
    return (clusterSets,centers,distortion)


def prepInputsKMeansPlusPlus():
    data = []
    clusters = {} 
    clusterSets = {}
    centers = []
    with open("toydata.txt") as f:
        for line in f:
            lineSplit = line.split()
            data.append((float(lineSplit[0]), float(lineSplit[1])))
            clusters[(float(lineSplit[0]),float(lineSplit[1]))] = -1
    firstSelect = random.randint(0,len(data)-1)
    centers.append(data[firstSelect])
    probabilities = []
    minDists = []
    for i in range(0,len(data)):
        minDist = sys.maxint 
        denom = 1
        for centroid in centers:
            if(dist(data[i],centroid)) <= minDist:
                minDist = dist(data[i],centroid)
        minDists.append(minDist)
    sumMinDists = 0
    for item in minDists:
        sumMinDists += item**2
    for i in range(0,len(data)):
        probabilities.append(minDists[i]**2/sumMinDists)
    for j in range(1,K):
        choice = np.random.choice(np.array(len(data)),1,p=probabilities)
        centers.append(data[choice])
    return (data,clusters,clusterSets,centers)


def kmeansplusplus():
    data,clusters,clusterSets,centers = prepInputsKMeansPlusPlus()
    clusters,centers,distortion = kmeans(data,clusters,clusterSets,centers)
    return distortion



for i in range(20):
    distortion_KMeans = kmeansplusplus()
    distortion_KMeansPlusPlus = kmeansplusplus()
    print distortion
    with open("clusters.csv", "ab") as output:
        out = csv.writer(output, dialect='excel')
        out.writerow((str(i),''))
        for item in distortion:
            out.writerow((str(item),''))
#--------This code used to print out the 2D graph of the clusters---------
#	for key in clusters:
#            out.writerow((str(key),''))
#            for pair in clusters[key]:
#                out.writerow((str(pair[0]),str(pair[1])))

