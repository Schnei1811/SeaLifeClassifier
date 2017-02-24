import scipy
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import os
from sklearn.cluster import KMeans
import time
import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
np.set_printoptions(threshold=1000)


def median_pixel_value():
    # To prevent ludicrous array sizes, median calculated in batches. Median of batches calculated
    print('\nDetermining Median Values\n')
    if not os.path.exists('Files/PixelAverages/'.format(videoname)): os.makedirs('Files/PixelAverages/'.format(videoname))
    selectedframes = np.zeros(framesconsidered)
    medianvalues = np.zeros(width * height)
    intermediatemedianvalues = np.zeros(width * height)
    finalmedianvalues = np.zeros(width * height)
    #To increase running time, reduce number of frames considered.
    for i in range(1, framesconsidered): selectedframes[i] = (maxnumframes/framesconsidered) + selectedframes[i-1]
    for framenum in tqdm.trange(0, framesconsidered):
        ret, frame = mediancapture.read()
        mediancapture.set(1, selectedframes[framenum])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()
        if framenum == 0: data = gray
        else: data = np.vstack((data, gray))
        if framenum == medianbatchsize:
            for i, j in enumerate(gray): intermediatemedianvalues[i] = (int(np.median(data[:, i])))
            data = gray
        elif framenum > medianbatchsize and framenum % medianbatchsize == 0:
            for i, j in enumerate(gray): medianvalues[i] = (int(np.median(data[:, i])))
            data = gray
            if framenum > medianbatchsize and framenum < framesconsidered:
                intermediatemedianvalues = np.vstack((intermediatemedianvalues, medianvalues))
        if framenum == framesconsidered - 1:
            for i, j in enumerate(gray):
                finalmedianvalues[i] = (int(np.median(intermediatemedianvalues[:, i])))
            np.savetxt('Files/PixelAverages/PixelMedianData-{}.txt'.format(videoname), finalmedianvalues, fmt='%i', delimiter=',')
    print('\n{} Median Values Calculated \n'.format(videoname))
    return finalmedianvalues

def identifyboxes(medianvalues):
    print('\nIdentifying Boxes Within Video')
    if not os.path.exists('Files/{}/'.format(videoname)): os.makedirs('Files/{}/'.format(videoname))

    for framenum in tqdm.trange(0, maxnumframes):
        ret, frame = boxcreatecapture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()

        #Compare Pixel Value to Threshold. (Longest running time 0.85)
        thresholdvalues = (medianvalues - gray)
        for i, j in enumerate(thresholdvalues):
            if abs(thresholdvalues[i]) > threshold: gray[i] = 255       #pixel threshold
            else: gray[i] = 0

        gray = gray.reshape(height, width)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.threshold(blurred, blur, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        labels = measure.label(thresh, neighbors=8, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        # Determining Contours.  2nd Longest Running time 0.27
        for label in np.unique(labels):
             if label == 0: continue
             labelMask = np.zeros(thresh.shape, dtype="uint8")
             labelMask[labels == label] = 255
             numPixels = cv2.countNonZero(labelMask)
             if numPixels > 100: mask = cv2.add(mask, labelMask)
        ###

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]

        # Create Square around Contours
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)              
            newsquare = np.array([x - squaresize, y - squaresize, x + w + squaresize, y + h + squaresize], ndmin=2)
            if i == 0: squarestorage = newsquare
            else: squarestorage = np.vstack((squarestorage, newsquare))

        # Create Adjacency Matrix
        intersectiongraph = np.zeros((len(squarestorage), len(squarestorage)))
        for i in range(0, len(squarestorage)):
            for j in range(0, len(squarestorage)):
                if i == j: intersectiongraph[i, j] = 0
                elif (squarestorage[i, 2] < squarestorage[j, 0] or squarestorage[j, 2] < squarestorage[i, 0]
                      or squarestorage[i, 3] < squarestorage[j, 1] or squarestorage[j, 3] < squarestorage[i, 1]):
                    intersectiongraph[i, j] = 0
                else: intersectiongraph[i, j] = 1

        # Determine Strongly Connected Components
        uniquesquares = np.array([scipy.sparse.csgraph.connected_components(intersectiongraph, directed=False, connection='weak', return_labels = True)[1]])
        squarestorage = np.concatenate((uniquesquares.T, squarestorage), axis=1)
        squarestorage = squarestorage[np.argsort(squarestorage[:, 0])]
        finalsquares = np.zeros((np.max(uniquesquares), 5))

        # Determine Min/Max of SCCs
        newsquareiter = 0
        squarecount = 0
        for i, j in enumerate(squarestorage):
            if squarestorage[i, 0] != squarecount:
                finalsquares[squarecount, 0] = squarecount
                if np.amin([squarestorage[newsquareiter:i, 1]]) < 0:
                    finalsquares[squarecount, 1] = 0
                else: finalsquares[squarecount, 1] = np.amin([squarestorage[newsquareiter:i, 1]])
                if np.amin([squarestorage[newsquareiter:i, 2]]) < 0:
                    finalsquares[squarecount, 2] = 0
                else: finalsquares[squarecount, 2] = np.amin([squarestorage[newsquareiter:i, 2]])
                if np.amax([squarestorage[newsquareiter:i, 3]]) > width:
                    finalsquares[squarecount, 3] = width
                else: finalsquares[squarecount, 3] = np.amax([squarestorage[newsquareiter:i, 3]])
                if np.amax([squarestorage[newsquareiter:i, 4]]) > height:
                    finalsquares[squarecount, 4] = height
                else: finalsquares[squarecount, 4] = np.amax([squarestorage[newsquareiter:i, 4]])
                squarecount += 1
                newsquareiter = i

        finalsquares = finalsquares.astype(int)
        framenumlist = np.array([np.ones((len(finalsquares),), dtype=np.int)*framenum])
        finalsquares = np.concatenate((framenumlist.T, finalsquares), axis=1)

        if framenum == 0: completevideofinalsquares = finalsquares
        else: completevideofinalsquares = np.vstack((completevideofinalsquares, finalsquares))

        if framenum % savestate == 0:
            np.savetxt('Files/{}/temp{}{}{}VideoBoxesArray.txt'.format(videoname, threshold, blur, squaresize), completevideofinalsquares, fmt='%i', delimiter=',')
        if framenum == maxnumframes-1:
            np.savetxt('Files/{}/{}{}{}VideoBoxesArray.txt'.format(videoname, threshold, blur, squaresize), completevideofinalsquares, fmt='%i', delimiter=',')
            os.remove('Files/{}/temp{}{}{}VideoBoxesArray.txt'.format(videoname, threshold, blur, squaresize))
    return completevideofinalsquares

def boxpixeldata(boxdata):
    print('\nGathering Pixel Data of Boxes')
    roiw, roih = uniformboxsize, uniformboxsize
    finalsquarearray = np.zeros([roiw * roih])

    # Resize Boxes a place in Array for Unsupervised Learning
    for framenum in tqdm.trange(0, maxnumframes):
        newframecounter = 0
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i, j in enumerate(boxdata):
            if boxdata[i, 0] == framenum and (boxdata[i, 5] - boxdata[i, 3]) * (boxdata[i, 4] - boxdata[i, 2]) > 10000:
                squarearray, squarecounter = np.zeros([roiw * roih]), 0
                roi = gray[boxdata[i, 3]:boxdata[i, 5], boxdata[i, 2]:boxdata[i, 4]]
                roi = cv2.resize(roi, (roiw, roih))
                for y in range(0, roih):
                    for x in range(0, roiw):
                        squarearray[squarecounter] = roi[y, x]  # works [height,width]
                        squarecounter += 1
                if newframecounter == 0:
                    intermediatesquarearray = squarearray
                    newframecounter += 1
                else: intermediatesquarearray = np.vstack((intermediatesquarearray, squarearray))
        if framenum == 0: finalsquarearray = intermediatesquarearray
        else: finalsquarearray = np.vstack((finalsquarearray, intermediatesquarearray))

        if framenum % savestate == 0:
            np.savetxt('Files/{}/temp{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize)
                       , finalsquarearray, fmt='%i', delimiter=',')
        if framenum == maxnumframes - 1:
            np.savetxt('Files/{}/{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize)
                       , finalsquarearray, fmt='%i', delimiter=',')
            os.remove('Files/{}/temp{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize))
        framenum += 1
    return finalsquarearray

def KMeansfunc(squares):
    print('\nRunning KMeans on Identified Boxes')
    clf = KMeans(n_clusters=numclusters)
    clf.fit(squares)
    labels = clf.labels_
    np.savetxt('Files/{}/{}x{}Labels.txt'.format(videoname, uniformboxsize, uniformboxsize), labels, fmt='%i', delimiter=',')
    return labels

def saveimages(labels, imgdata):
    print('\nSaving Images')
    if not os.path.exists('Files/{}/Clusters/'.format(videoname)): os.makedirs('Files/{}/Clusters'.format(videoname))
    for dir in tqdm.trange(0, numclusters):
        if not os.path.exists('Files/{}/Clusters/{}'.format(videoname, dir)): os.makedirs('Files/{}/Clusters/{}'.format(videoname, dir))
        for i, j in enumerate(imgdata):
            if labels[i] == dir:
                roi = imgdata[i, :].reshape((uniformboxsize, uniformboxsize))
                cv2.imwrite('Files/{}/Clusters/{}/{}{}.jpg'.format(videoname, dir, i, videoname), roi)
                img = cv2.imread('Files/{}/Clusters/{}/{}{}.jpg'.format(videoname, dir, i, videoname))
                img = cv2.resize(img, (200, 200))
                cv2.imwrite('Files/{}/Clusters/{}/{}{}.jpg'.format(videoname, dir, i, videoname), img)

def ConvNet(boxdata):
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, classifications, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model Loaded!')
    else:
        print('Error Loading Model')
        return

    convroiw, convroih = IMG_SIZE, IMG_SIZE
    boxcounter = 0

    # Classify Boxes
    for framenum in tqdm.trange(0, maxnumframes):
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i, j in enumerate(boxdata):
            if boxdata[i, 0] == framenum and (boxdata[i, 5] - boxdata[i, 3]) * (boxdata[i, 4] - boxdata[i, 2]) > 8000:
                roi = gray[boxdata[i, 3]: boxdata[i, 5], boxdata[i, 2]: boxdata[i, 4]]
                roi = cv2.resize(roi, (convroiw, convroih)).reshape(convroiw, convroih, 1)
                model_out = model.predict([roi])
                boxid = np.insert(boxdata[i, :], 6, np.argmax(model_out))
                if boxcounter == 0:
                    data = boxid
                    boxcounter += 1
                else: data = np.vstack((data, boxid))
    np.savetxt('Files/{}/{}{}{}VideoBoxID.txt'.format(videoname, threshold, blur, squaresize), data, fmt='%i', delimiter=',')

#videoname = '1chargingbehaviour'
#videoname = '2corrallingbehaviour'
videoname = '3corrallingfromanotherangle'
#videoname = '4successfulescape'
#videoname = '5ShortClip'
#videoname = 'test'
#videoname = 'reddit'
#videoname = 'GO010141'

mediancapture = cv2.VideoCapture('Files/Videos/{}.mp4'.format(videoname))
boxcreatecapture = cv2.VideoCapture('Files/Videos/{}.mp4'.format(videoname))
capture = cv2.VideoCapture('Files/Videos/{}.mp4'.format(videoname))
maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = int(capture.get(3)), int(capture.get(4))
#framesconsidered = int(maxnumframes - (maxnumframes % 10))         #For short testing videos
framesconsidered = int(maxnumframes/4 - (maxnumframes/4 % 100))
initialtime = time.time()
print('\nVideo Name: ', videoname)
print('Total Frames: ', maxnumframes, '\n')

#PARAMETERS

#DETERMINE MEDIAN
medianbatchsize = 100

#CREATING BOXES
threshold = 8        # Lower = less difference from mean value
blur = 120           # Under blurring value, pixel cut off
squaresize = 8       # Square Sizes
savestate = 100

#STANDARDIZEd BOX SIZE FOR UNSUPERVISED LEARNING
uniformboxsize = 100

#KMEANS K VALUE
numclusters = 50

#CONVNET PARAMETERS
#tensorboard --logdir=foo:C:\Users\Stefan\Dropbox\PythonScripts\PythonScripts\opencv\Octopus\Files\log
classifications = 4
LR = 1e-3
IMG_SIZE = 50
MODEL_NAME = '{}-{}.model'.format(LR, '8conv-basic')


if os.path.exists('Files/PixelAverages/PixelMedianData-{}.txt'.format(videoname)) == False:
    medianvalues = median_pixel_value()

if os.path.exists('Files/{}/{}{}{}VideoBoxesArray.txt'.format(videoname, threshold, blur, squaresize)) == False:
    medianvalues = np.loadtxt('Files/PixelAverages/PixelMedianData-{}.txt'.format(videoname), delimiter=',')
    boxdata = identifyboxes(medianvalues)

if os.path.exists('Files/{}/{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize)) == False:
    boxdata = np.loadtxt('Files/{}/{}{}{}VideoBoxesArray.txt'.format(videoname, threshold, blur, squaresize), delimiter=',')
    unsupervisedlearningdata = boxpixeldata(boxdata)

if os.path.exists('Files/{}/{}x{}Labels.txt'.format(videoname, uniformboxsize, uniformboxsize)) == False:
    unsupervisedlearningdata = np.loadtxt('Files/{}/{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize), delimiter=',')
    labels = KMeansfunc(unsupervisedlearningdata)

if os.path.exists('Files/{}/Clusters/'.format(videoname)) == False:
    unsupervisedlearningdata = np.loadtxt('Files/{}/{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize), delimiter=',')
    labels = np.loadtxt('Files/{}/{}x{}Labels.txt'.format(videoname, uniformboxsize, uniformboxsize), delimiter=',')
    saveimages(labels, unsupervisedlearningdata)

if os.path.exists('Files/{}/{}{}{}VideoBoxID.txt'.format(videoname, threshold, blur, squaresize)) == False:
    boxdata = np.loadtxt('Files/{}/{}{}{}VideoBoxesArray.txt'.format(videoname, threshold, blur, squaresize), delimiter=',').astype(int)
    ConvNet(boxdata)

print('Complete!!')

# historic box values:
# (50 175 60)   Squares too wide
# (50 175 50)   Doesn't detect enough
# (30 160 50)   Detects more. Squares too wide
# (30 160 35)   Needs to detect more. Squares too wide. Not many false positives
# (30 160 25)   Squares still too big. Rare false positives. Still needs to detect slightly more
# (0 140 20)    Terrible. False positives. Disconnected Squares
# (20 160 20)   Like Box Size
# (20 150 20)
# (15 150 15)   Like Threshold. Box Size Slightly too small.
# (15 150 20)   Misses some moving objects
# (15 130 20)   Better with Objects. Squares too big
# (10 130 20)   Good. Box sizes too big
# (10 130 15)   Seems great. Creating Copies. Boxes too big.
# (10 130 8)    Current Version. Sometimes still need greater connectivity...
# (8 120 8)

# cv2.rectangle(gray, (squares[i, 2], squares[i, 3]), (squares[i,4], squares[i,5]), (0, 255, 0), 2)
# print(squarearray)
# cv2.imshow("Image", roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
