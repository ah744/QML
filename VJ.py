#!/usr/bin/python

from PIL import Image, ImageDraw 
from matplotlib import pyplot as plt
import sys 
import numpy as np

iimages = []
featuretbl = []
NUM_EXAMPLES = 50 

def readTraining():
    trainingFaces = []
    trainingBackgrounds= []
    for i in range(NUM_EXAMPLES/2):
        face = np.asarray(Image.open('faces/face' + str(i) + '.jpg','r').convert('L'), dtype=np.int32)
        backg = np.asarray(Image.open('background/'+str(i) + '.jpg','r').convert('L'), dtype=np.int32)
        trainingFaces.append(face)
        trainingBackgrounds.append(backg)
    return trainingFaces+trainingBackgrounds

def fillFeatures():
    # Feature type 1
    for col1 in range(0,64,8):
        for row1 in range(0,64,8):
            for col2 in range(col1+4,64, 1):
                for row2 in range(row1+4,64, 1):
                    if col2+(col2-col1) < 64:
                        featuretbl.append([(row1,col1),(row2,col2),(row1,col2+1),(row2,col2+(col2-col1))])
    # Feature type 2
    for col1 in range(0,64,8):
        for row1 in range(0,64,8):
            for col2 in range(col1+4,64, 1):
                for row2 in range(row1+4,64, 1):
                    if row2+(row2-row1) < 64:
                        featuretbl.append([(row2+1,col1),(row2+(row2-row1),col2),(row1,col1),(row2,col2)])

def calcIntegralImages(images):
    for image in images:
        iimages.append(image.cumsum(axis=0).cumsum(axis=1))

def computeFeature(image,f):
    ULL = f[0]
    BRL = f[1]
    ULD = f[2]
    BRD = f[3]
    BLD = (BRD[0],ULD[1])
    URD = (ULD[0],BRD[1]) 
    BLL = (BRL[0],ULL[1])
    URL = (ULL[0],BRL[1])
    darkSum = image[BRD] + image[ULD] - image[URD] - image[BLD]
    lightSum = image[BRL] + image[ULL] - image[URL] - image[BLL]
    return darkSum - lightSum

def correctLabel(example):
    if example >= NUM_EXAMPLES/2:
        return 0
    return 1

def setPolarityThreshold(feature,weights,examples, T_, TP):
    scores = []
    for i in range(len(examples)): 
        scores.append((computeFeature(examples[i],feature),i)) 
    scores.sort(key=lambda x: x[0]) 
    errors = []
#    T_ = sum([weights[pair[1]] for pair in scores if correctLabel(pair[1]) == 0]) 
#    TP = sum([weights[pair[1]] for pair in scores if correctLabel(pair[1]) == 1]) 
    SSP = np.array([weights[pair[1]] if correctLabel(pair[1]) == 1 else 0 for pair in scores]).cumsum()
    SS_ = np.array([weights[pair[1]] if correctLabel(pair[1]) == 0 else 0 for pair in scores]).cumsum()
    for i in range(len(scores)): 
        polarity = 1
        S_ = SS_[i]-weights[i] 
        SP = SSP[i]-weights[i] 
        left = SP+(T_-S_) 
        right = S_+(TP-SP) 
        error = min(left,right)
        if error == left:
            polarity = -1
        errors.append((error, scores[i][0], polarity))
    return min(errors, key=lambda x:x[0])[1:] 
         
def predict(image, feature, threshold, polarity):
    fx = computeFeature(image,feature)
    if polarity * fx < polarity * threshold:
        return 1
    return 0

def bestLearner(weights,examples):
    minError = sys.maxint
    learner = (0,0,0,0)
    T_ = sum([weights[i] for i in range(len(examples)) if correctLabel(i) == 0]) 
    TP = sum([weights[i] for i in range(len(examples)) if correctLabel(i) == 1]) 
    for j in range(len(featuretbl)):
        error = 0
        thresh,pol = setPolarityThreshold(featuretbl[j], weights,examples,T_,TP)
        for i in range(len(examples)):
            error += weights[i] * abs(predict(examples[i],featuretbl[j],thresh,pol) - correctLabel(i))
        if error <= minError:
            minError = error
            beta = error/(1-error)
            learner = (j,pol,thresh,beta)
    return learner 

def updateWeights(learner, weights, examples):
    for i in range(len(examples)):
        if predict(examples[i], featuretbl[learner[0]], learner[2], learner[1]) == correctLabel(i):
            weights[i] = weights[i]*learner[3]
    return weights

def setBigTheta(classifier,examples):
    bigTheta = 0
    for i in range(len(examples)):
        value = 0
        threshold = 0
        for classifierPair in classifier:
            fullPredict = 0 
            learner = classifierPair[0]
            alpha = classifierPair[1]
            prediction = predict(examples[i],featuretbl[learner[0]],learner[2],learner[1])
            value += alpha*prediction 
            threshold += alpha 
        if value >= threshold/2:
            fullPredict = 1
        elif value < threshold/2 and correctLabel(i) == 1:
            bigTheta = max(bigTheta, (threshold/2-value))
    return bigTheta

def fullPrediction(image,classifierSet,bigTheta):
    value = 0
    threshold = 0
    for classifierPair in classifierSet:
        learner = classifierPair[0]
        alpha = classifierPair[1]
        prediction = predict(image,featuretbl[learner[0]],learner[2],learner[1])
        value += alpha*prediction 
        threshold += alpha 
    if value + bigTheta >= threshold/2:
        return 1
    return 0

def calcFalseRates(classifierSet,bigTheta,examples):
    falsePositives = 0.
    falseNegatives = 0.
    negatives = 0.
    positives = 0.
    predictions = []
    for i in range(len(examples)):
        prediction = fullPrediction(examples[i],classifierSet,bigTheta)
        predictions.append(prediction)
        if correctLabel(i) == 0:
            negatives += 1
        elif correctLabel(i) == 1:
            positives += 1
        if prediction == 1 and correctLabel(i) == 0:
            falsePositives += 1
        elif prediction == 0 and correctLabel(i) == 1:
            falseNegatives += 1
    print "False Positives, Backgrounds:"
    print falsePositives,negatives
    print "False Negatives, Faces:"
    print falseNegatives,positives
    return falsePositives/negatives, predictions

def adaBoost(examples,numPositives):
    #Init weights
    w_p = 1./(2*numPositives)
    w_n = 1./(2*(len(examples)-numPositives))
    print w_p,w_n
    print "Weights initialized:"
    w = [w_p if i < numPositives else w_n for i in range(len(examples))] 
    classifierSet = []
    bigTheta = 0
    t = 0
    T = 1
    while t < T:
        #normalize weights
        w = w / np.sum(w)
        bigTheta = 0
        learner = bestLearner(w,examples)
        print learner
        alpha = 0
        if learner[3] == 0:
            alpha = 1
        else:
            alpha = np.log(1/learner[3])
        w = updateWeights(learner,w,examples)
        print w
        classifierSet.append((learner,alpha))
        bigTheta = setBigTheta(classifierSet,examples)
        print "Big Theta: " 
        print bigTheta
        fP,predictions = calcFalseRates(classifierSet,bigTheta,examples)
        if fP > 0.3:
            T += 1
        t += 1
    return (classifierSet,bigTheta,predictions)

def train(examples):
    newTrainingSet = examples[:]
    numPositives = NUM_EXAMPLES/2
    classifierSet = []
    stopValue = float(len(examples))
    print len(newTrainingSet),stopValue
    print len(newTrainingSet)/stopValue
    while float(len(newTrainingSet))/stopValue >= 0.51:
        print "New training stage: " + str(len(newTrainingSet))
        classifier,bigTheta,predictions = adaBoost(newTrainingSet,numPositives)
        newTrainingSet = [examples[i] for i in range(len(predictions)) if predictions[i] == 1]
        classifierSet.append((classifier,bigTheta))
    return classifierSet 

def readTest():
    return np.asarray(Image.open('class.jpg','r').convert('L'), dtype=np.int32).cumsum(axis=0).cumsum(axis=1)

def generateWindows(image):
    windows = []
    coords = []
    print image.shape
    for top_left_row in range(0,image.shape[0],8):
        for top_left_col in range(0,image.shape[1],8): 
            bottom_right = (top_left_row + 64,top_left_col+64)
            if bottom_right[0] < image.shape[0] and bottom_right[1] < image.shape[1]:
                window = image[np.ix_([i for i in np.arange(top_left_row,bottom_right[0])],[j for j in np.arange(top_left_col,bottom_right[1])])]
                windows.append(window)
                coords.append(((top_left_col,top_left_row),(bottom_right[1],bottom_right[0])))
    print coords[0], coords[-1]
    return windows,coords
            
def classifierCascade(cSet, windows, coords):
    print cSet
    predictedFaces = []
    for classifier in cSet: 
        numFaces = 0
        numBacks = 0
        predictedFaces = []
        newWindows = []
        for i in range(len(windows)): 
            prediction = fullPrediction(windows[i],classifier[0],classifier[1])
            if prediction == 0:
                numBacks += 1
            else:
                numFaces += 1
                predictedFaces.append(coords[i])
                newWindows.append(windows[i])
        print "Predictions: Faces = " + str(numFaces) + " Backgrounds = " + str(numBacks)
        windows = newWindows[:]
        coords = predictedFaces[:]
    return predictedFaces

def overlap(coordsSet, coords2):
    for coords1 in coordsSet:
        upLeft = coords2[0]
        botRight = coords2[1]
        botLeft = (botRight[0],upLeft[1])
        upRight = (upLeft[0],botRight[1])
        points = [upLeft,botRight,botLeft,upRight]
        for point in points:
            if point[0] >= coords1[0][0] and point[0] <= coords1[1][0] and point[1] >= coords1[0][1] and point[1] <= coords1[1][1]:
                return True 
    return False 

def excludeOverlaps(faces):
    excludedFaces = []
    for image in faces:
        if not overlap(excludedFaces,image):
            excludedFaces.append(image)
    return excludedFaces

def drawFaces(faces):
    test = Image.open("class.jpg") 
    draw = ImageDraw.Draw(test)
    drawingFaces = excludeOverlaps(faces)
    for i in range(len(drawingFaces)):
        draw.rectangle(drawingFaces[i],fill=None,outline=(255))
    test.save("faces.jpg")
    test.show()

# Program Main #

training = np.array(readTraining())
test = readTest()
windows,coords = generateWindows(test)
calcIntegralImages(training)
fillFeatures()
classifierCascadeSet = train(iimages)
faces = classifierCascade(classifierCascadeSet, windows, coords) 
drawFaces(faces)
