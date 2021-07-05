import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import preprocess

# Return a list of all image names with a given extension in a given folder
def listImages(dir, extension):
    res = []
    for img in os.listdir(dir):
        if img.endswith(extension):
            res.append(img)
    return res

# Return a list of all image names in subfolders with a given extension in a given folder
def listImagesSub(dir, extension):
    res = []
    for subdir in os.listdir(dir):
        for img in os.listdir(dir + subdir):
            if img.endswith(extension):
                res.append(subdir + "/" + img)
    return res

# Compute the slope of a given line
# Infite slopes (vertical lines) are set to the height of the image
def getSlope(line, height):
    # line is [x1, y1, x2, y2]
    if (line[2] - line[0]) == 0:
        slope = height
    else:
        slope = ((line[3]-line[1]) / (line[2]-line[0]))
    return slope

# Calculate perpendicular slope
# Return true if two slopes are perpendicular
# Will also return true is slopes are close to perpendicular (within a range of 0-1)
def isPerpendicular(slope1, slope2):
    if slope1 != 0:
        if (abs((-1 * (1 / slope1)) - slope2) < 1):
            return True
    elif slope2 != 0:
        if (abs((-1 * (1 / slope2)) - slope1) < 1):
            return True
    return False


# Calculate intersection between two lines, return None if no intersection
# from https://rosettacode.org/wiki/Find_the_intersection_of_two_lines#Python
def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):

    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
 
    return x, y

# Calculate the distance of a point to all endpoints of two lines
# returns minimum of these distances
# Used to calculate radius of rotaional symmetry
# Minimum distance from intersection to any endpoint = radius
def minDistance(intersect, line1, line2):

    dist1 = np.sqrt( (line1[0] - intersect[0])**2 + (line1[1] - intersect[1])**2 )
    dist2 = np.sqrt( (line1[2] - intersect[0])**2 + (line1[3] - intersect[1])**2 )
    dist3 = np.sqrt( (line2[0] - intersect[0])**2 + (line2[1] - intersect[1])**2 )
    dist4 = np.sqrt( (line2[2] - intersect[0])**2 + (line2[3] - intersect[1])**2 )

    return (min(dist1, dist2, dist3, dist4))

# Used to reorder symmetries for ease of processing 
# from:
# [[[[[line], slope, score, normScore, depth], ...], depth] ... ]
# To:
# [[[line], slope, score, normScore, depth], ...]
def placeInOrder(symmetries):
    newSymmetries = []
    for syms in symmetries:
        newSymmetries += syms[0]
    return newSymmetries

# Only required when cuts are made before knowing symThreshold (Ipynb kernel)
# Will also reorder the symmetries as placeInOrder does
# Will remove all smaller parts of an image where the cut was made on a reflection 
# symmetry line with a low score. Will also remove all parts based on that recursive loop.
def removeBadCuts(symmetries, symThreshold):
    newSymmetries = []
    deleteDepth = 99999
    for syms in symmetries:
        if syms[1] == 0:
            newSymmetries += syms[0]
            continue
        if syms[1] >= deleteDepth:
            symmetries.remove(syms)
            continue
        else:
            deleteDepth = 99999
            mainSym = syms[0][0]
            if mainSym[2] < symThreshold:
                deleteDepth = syms[1] + 1
                symmetries.remove(syms)
                continue
            else:
                newSymmetries += syms[0]
    return newSymmetries

# Remove symmetries if they have a normalized score under normThreshold
# or if they have a normalized score of 1.0 and a score under symThreshold, i.e. are the main symmetry in their recursive loop (sub image)
# If a main symmetry is removed, all other symmetries in that recursive loop are also removed, 
# by removing next symmetries untill they have a different depth, meaning they belong to a different loop
def removeBadSymmetries(symmetries, symThreshold, normThreshold):
    copySym = symmetries[:]

    for i in range(0, len(symmetries)):
        if symmetries[i] not in copySym:
            continue
        if symmetries[i][3] < normThreshold:
            copySym.remove(symmetries[i])
        elif symmetries[i][3] == 1.0:
            if symmetries[i][2] < symThreshold:
                copySym.remove(symmetries[i])
                j = i + 1
                if j >= len(symmetries):
                    break
                while (symmetries[i][4] == symmetries[j][4]):
                    copySym.remove(symmetries[j])
                    j = j + 1
                    if j >= len(symmetries):
                        break

    return copySym

# Loop over each line and compare to other lines
# If slope is similar and the distance between endpoints is small enough, remove line with lower symmetry score
# maxDistX and maxDistY set based on width and height of image
# Both dictate the maximum distance between endpoints of lines
# If line1-endpoint1 is within maxDist to line2-endpoint1
# line1-endpoint2 only has to lie within (maxDist*0,66) line2-endpoint2 to be flagged as similar
def removeSimilarLines(symmetries, image, lineSimilarity):

    height, width, _ = image.shape

    maxDistX = width / lineSimilarity
    maxDistY = height / lineSimilarity
    maxDist = (maxDistX + maxDistY) / 2
    maxSlopeDiff = maxDistY

    copySym = symmetries[:]

    def lowerScore(sym1, sym2):
        if sym1[2] < sym2[2]:
            return sym1
        return sym2

    for i in range(0, len(copySym)):
        for j in range(i + 1, len(copySym)):
            if copySym[i] not in symmetries:
                break
            if copySym[j] not in symmetries:
                continue
            if abs(copySym[i][1] - copySym[j][1]) < maxSlopeDiff or (abs(copySym[i][1]) > height / 3 and abs(copySym[j][1]) > height / 3):
                dist = np.sqrt( (copySym[i][0][0] - copySym[j][0][0])**2 + (copySym[i][0][1] - copySym[j][0][1])**2 )
                if dist < maxDist:
                    dist = np.sqrt( (copySym[i][0][2] - copySym[j][0][2])**2 + (copySym[i][0][3] - copySym[j][0][3])**2 )
                    if dist < maxDist * (lineSimilarity * 0.66):
                        symmetries.remove(lowerScore(copySym[i], copySym[j]))
                        continue
                
                dist = np.sqrt( (copySym[i][0][0] - copySym[j][0][2])**2 + (copySym[i][0][1] - copySym[j][0][3])**2 )
                if dist < maxDist:
                    dist = np.sqrt( (copySym[i][0][2] - copySym[j][0][0])**2 + (copySym[i][0][3] - copySym[j][0][1])**2 )
                    if dist < maxDist * (lineSimilarity * 0.66):
                        symmetries.remove(lowerScore(copySym[i], copySym[j]))
                        continue

                dist = np.sqrt( (copySym[i][0][2] - copySym[j][0][0])**2 + (copySym[i][0][3] - copySym[j][0][1])**2 )
                if dist < maxDist:
                    dist = np.sqrt( (copySym[i][0][0] - copySym[j][0][2])**2 + (copySym[i][0][1] - copySym[j][0][3])**2 )
                    if dist < maxDist * (lineSimilarity * 0.66):
                        symmetries.remove(lowerScore(copySym[i], copySym[j]))
                        continue
                
                dist = np.sqrt( (copySym[i][0][2] - copySym[j][0][2])**2 + (copySym[i][0][3] - copySym[j][0][3])**2 )
                if dist < maxDist:
                    dist = np.sqrt( (copySym[i][0][0] - copySym[j][0][0])**2 + (copySym[i][0][0] - copySym[j][0][1])**2 )
                    if dist < maxDist * (lineSimilarity * 0.66):
                        symmetries.remove(lowerScore(copySym[i], copySym[j]))
                        continue
    return symmetries

# Remove similar rotational symmetries
# Remove if centerpoint is within maxDistX and maxDistY and the radius is within max(maxDistX, maxDistY)
# Rotation symmetry which has the highest avarage depth is removed
# Average depth is calculated based on the depth of the two reflection lines that form the rotational symmetry
def removeSimilarRotational(rotations, image, rotationSimilarity):
    height, width, _ = image.shape

    maxDistX = width / rotationSimilarity
    maxDistY = height / rotationSimilarity
    copyRot = rotations[:]

    def higherDepth(rot1, rot2):
        if rot1[2] > rot2[2]:
            return rot1
        return rot2

    for i in range(0, len(copyRot)):
        for j in range(i + 1, len(copyRot)):
            if copyRot[i] not in rotations:
                break
            if copyRot[j] not in rotations:
                continue
            if abs(copyRot[i][0][0] - copyRot[j][0][0]) < maxDistX:
                if abs(copyRot[i][0][1] - copyRot[j][0][1]) < maxDistY:
                    if abs(copyRot[i][1] - copyRot[j][1]) < max(maxDistX, maxDistY):
                        rotations.remove(higherDepth(copyRot[i], copyRot[j]))

# Checks if distance between intersection point and endpoints of reflection lines is similar enough
# Used to calculate rotational symmetries with a non ML approach
def checkDistance(intersect, line1, line2, distDifference):

    dist1 = np.sqrt( (line1[0] - intersect[0])**2 + (line1[1] - intersect[1])**2 )
    dist2 = np.sqrt( (line1[2] - intersect[0])**2 + (line1[3] - intersect[1])**2 )
    dist3 = np.sqrt( (line2[0] - intersect[0])**2 + (line2[1] - intersect[1])**2 )
    dist4 = np.sqrt( (line2[2] - intersect[0])**2 + (line2[3] - intersect[1])**2 )

    if abs(dist1 - dist2) > distDifference:
        return False
    elif abs(dist1 - dist3) > distDifference:
        return False
    elif abs(dist1 - dist4) > distDifference:
        return False
    elif abs(dist2 - dist3) > distDifference:
        return False
    elif abs(dist2 - dist4) > distDifference:
        return False
    elif abs(dist3 - dist4) > distDifference:
        return False
    return True

# Find rotaional symmetries with a given machine learning model
# Will loop over each reflection symmetry in a double loop and check if any have intersections
# Pairs with intersections will be pre-processed and subsequently predicted by the model
# Positive results will create a rotational symmetry in their centerpoint
# The radius is determined by the minDistance function
# Reflection symmetries which create a rotational symmetrie are removed afterwards
# Will not be executed in 'fast' mode
def rotationalSymmetriesML(symmetries, model, data):
    h, w, _ = data.shape
    rotations = []
    tmp = []
    for i in range(0, len(symmetries)):
        for j in range(i + 1, len(symmetries)):
            intersect = line_intersect(symmetries[i][0][0], symmetries[i][0][1], symmetries[i][0][2], symmetries[i][0][3], symmetries[j][0][0], symmetries[j][0][1], symmetries[j][0][2], symmetries[j][0][3])
            if intersect == None:
                continue
            s = pd.Series(data={
                "line1x1": symmetries[i][0][0],
                "line1y1": symmetries[i][0][1],
                "line1x2": symmetries[i][0][2],
                "line1y2": symmetries[i][0][3],
                "line1Score": symmetries[i][2],
                "line2x1": symmetries[j][0][0],
                "line2y1": symmetries[j][0][1],
                "line2x2": symmetries[j][0][2],
                "line2y2": symmetries[j][0][3],
                "line2Score": symmetries[j][2],
                "height": h,
                "width": w
            }, name="rotation")
            data = pd.DataFrame()
            data = data.append(s, ignore_index=False)
            
            data = preprocess.preproccesData(data)
            pred = model.predict(data)
            
            if pred == True:
                rad = minDistance(intersect, symmetries[i][0], symmetries[j][0])
                meanDepth = (symmetries[i][4] + symmetries[j][4]) / 2
                rot = [intersect, rad, meanDepth]
                rotations.append(rot)
                if symmetries[i] not in tmp:
                    tmp.append(symmetries[i])
                if symmetries[j] not in tmp:
                    tmp.append(symmetries[j])

    for t in tmp:
        symmetries.remove(t)
    return rotations

# Find rotaional symmetries given reflection symmetries and a threshold
# Will loop over each reflection symmetry in a double loop and check if any pairs:
# - have similar symmetry score, their relative score must be inside the circleSymThreshold 
# - have intersections,
# - are (close to) perpendicular
# - have distances from their endpoints to the intersection not too different from one another
# Positive results will create a rotational symmetry in their centerpoint
# The radius is determined by the minDistance function
# Reflection symmetries which create a rotational symmetrie are removed afterwards
# Will not be executed in 'slow' mode
def rotationalSymmetries(symmetries, image, circleSymThreshold):
    rotations = []
    tmp = []
    copySym = symmetries[:]
    height, width, _ = image.shape
    distDifference = min(height / 5, width / 5)
    for sym in symmetries:
        for subsym in copySym:
            # First check if lines have similar symmetry scores
            if max(sym[2], subsym[2]) * circleSymThreshold > min(sym[2], subsym[2]):
                continue
            # Check if lines are perpendicular
            if isPerpendicular(sym[1], subsym[1]) == False:
                continue
            intersect = line_intersect(sym[0][0], sym[0][1], sym[0][2], sym[0][3], subsym[0][0], subsym[0][1], subsym[0][2], subsym[0][3])
            if intersect != None:
                if checkDistance(intersect, sym[0], subsym[0], distDifference) == False:
                    continue
                rad = minDistance(intersect, sym[0], subsym[0])
                meanDepth = (sym[4] + subsym[4]) / 2
                rot = [intersect, rad, meanDepth]
                rotations.append(rot)
                if sym not in tmp:
                    tmp.append(sym)
                if subsym not in tmp:
                    tmp.append(subsym)
    # Remove symmetries that create rotational
    for t in tmp:
        symmetries.remove(t)
    return rotations

# Plot all given reflection symmetry lines
def plotLines(symmetries):
    n = 0
    for sym in symmetries:
        if sym[4] > n:
            n = sym[4]
    linewidth = 3
    # Colors dicated by colormap (default: viridis)
    colors = plt.cm.jet(np.linspace(0,1,n + 1))
    for i, sym in enumerate(symmetries):
        color = colors[sym[4]]
        x = [sym[0][0], sym[0][2]]
        y = [sym[0][1], sym[0][3]]
        plt.plot(x, y, color=color, linewidth=linewidth)

# Plot all given rotational symmetries
def plotRotations(rotations):
    for rot in rotations:
        circleSym = plt.Circle(rot[0], linewidth=2.5, radius=rot[1], color="yellow", fill=False)
        fig = plt.gcf()
        axs = fig.gca()
        axs.add_patch(circleSym)

# Used to resize an image by a given fraction
def resize_image(image, fraction):
    h, w, _ = image.shape
    desiredW = int(w / fraction)
    desiredH = int(h / fraction)
    dimensions = (desiredW, desiredH)
    resizedImage = cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)
    return resizedImage
    