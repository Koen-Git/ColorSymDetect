# Find rotational and reflection symmetries inside images in a given folder
# Uses https://github.com/mawady/ColorSymDetect to fetch reflection symmetries
# Usage:
# python main.py 
# arguements:
# --input "custom input folder" (default: ./input/)
# --output "custom output folder" (default: ./output/)
# --mode "slow/fast, either uses Machine learning (slow) to detect rotational symmetries or simple rules (fast)" (default: slow)
# Requires matlab to be installed with the python extension
# Tested on python 3.7.10 with matlab R2020B

import matlab.engine
from matplotlib import image
from matplotlib import pyplot as plt
import os
import util
import pickle
import parameters
import argparse

# Fetching parameters from parameters.py
# See parameters.py for detailed explaination of the parameters.
resize = parameters.resize
symThresholdBC = parameters.symThresholdBC
normThresholdBC = parameters.normThresholdBC
symThresholdAC = parameters.symThresholdAC
normThresholdAC = parameters.normThresholdAC
rc = parameters.rc
lineSimilarity = parameters.lineSimilarity
rotationSimilarity = parameters.rotationSimilarity
circleSymThreshold = parameters.circleSymThreshold

# Fetch reflection symmetry lines with Elewady's WaveletSym detection algorithm
# Return output is structured as follows:
# [[[x1, y1, x2, y2], slope, score, Normalized score, depth], ...]
# Depth is appended in the recursiveSym function
def getSymmetries(inputFile, data):
    symmetries = []
    symmetryList = eng.pySym(inputFile) #pySym = matlab file
    height, _, _ = data.shape
    for sym in symmetryList:
        line = [sym[0], sym[1], sym[2], sym[3]]
        slope = util.getSlope(line, height)
        score = sym[4]
        if len(sym) < 6:
            normScore = 1.0
        else:
            normScore = sym[5]
        newSym = [line, slope, score, normScore]
        symmetries.append(newSym)
    return symmetries

# Recursively cut up images and fetch symmetries until 
# the processed image is smaller than minSize parameter
def recursiveSym(img, symmetries, depth, minSize, rc, locMove={"h":0,"w":0}):
    h, w, _ = img.shape
    if h < minSize.get("h") or w < minSize.get("w"):
        return
    depth = depth + 1
    mat_a = matlab.uint8(img.tolist())
    syms = getSymmetries(mat_a, img)
    del mat_a
    if len(syms) < 1:
        return

    symThreshold = symThresholdBC
    if args.mode == "fast":
        symThreshold = symThresholdBC

    # Copy the top three symmetry lines (if they exist)
    # Any lines below threshold can be skipped, to reduce computation time
    mainSyms = []
    if rc > len(syms):
        rc = len(syms)
    for i in range(0, rc):
        if syms[i][2] > symThreshold:
            mainSyms.append(syms[i][0].copy())

    # Adjust location for symmetry lines in the cut images
    for sym in syms:
        sym[0][0] = sym[0][0] + locMove.get("w")
        sym[0][2] = sym[0][2] + locMove.get("w")
        sym[0][1] = sym[0][1] + locMove.get("h")
        sym[0][3] = sym[0][3] + locMove.get("h")
    
    # Appending depth for later processing
    for sym in syms:
        sym.append(depth)

    symmetries.append([syms, depth])

    del syms

    # For each top three symmetry line:
    # Cut the image in left / right or top / bottom half (diagonal lines are not considered) and process each image
    # newLocMove saves the symmetry line location adjustments for the next image
    for mainSym in mainSyms:
        if abs(mainSym[0] - mainSym[2]) < h/10:
            img01 = img[min(int(mainSym[1]), int(mainSym[3])):max(int(mainSym[1]), int(mainSym[3])), 0:int(mainSym[0])]
            img02 = img[min(int(mainSym[1]), int(mainSym[3])):max(int(mainSym[1]), int(mainSym[3])), int(mainSym[0]):w]

            newLocMove = {"h": locMove.get("h") + min(int(mainSym[1]), int(mainSym[3])), "w": locMove.get("w")}
            recursiveSym(img01, symmetries, depth, minSize, rc, newLocMove)

            newLocMove = {"h": locMove.get("h") + min(int(mainSym[1]), int(mainSym[3])), "w": locMove.get("w") + int(mainSym[0])}
            recursiveSym(img02, symmetries, depth, minSize, rc, newLocMove)
        elif abs(mainSym[1] - mainSym[3]) < w/10:
            img01 = img[0:int(mainSym[1]), min(int(mainSym[0]), int(mainSym[2])):max(int(mainSym[0]), int(mainSym[2]))]
            img02 = img[int(mainSym[1]):h, min(int(mainSym[0]), int(mainSym[2])):max(int(mainSym[0]), int(mainSym[2]))]

            newLocMove = {"h": locMove.get("h"), "w": locMove.get("w") + min(int(mainSym[0]), int(mainSym[2]))}
            recursiveSym(img01, symmetries, depth, minSize, rc, newLocMove)
            
            newLocMove = {"h": locMove.get("h") + int(mainSym[1]), "w": locMove.get("w") + min(int(mainSym[0]), int(mainSym[2]))}
            recursiveSym(img02, symmetries, depth, minSize, rc, newLocMove)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="slow", choices=["slow", "fast"], help="slow / fast (Machine learning turned on or off) (default: slow)")
    parser.add_argument("--input", default="./input/", help="Custom input folder (default: ./input/)")
    parser.add_argument("--output", default="./output/", help="Custom output folder (default: ./output/)")
    args = parser.parse_args()

    inDir = args.input
    outDir = args.output

    modelFileName = "models/RF-trainValTest.pkl"
    with open(modelFileName, 'rb') as file:
        model = pickle.load(file)
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    # Start matlab and fetch images
    eng = matlab.engine.start_matlab()
    imgList = util.listImages(inDir, '.jpg')
    images = []

    print(modelFileName)

    print("Fetching symmetries ...")

    for i, img in enumerate(imgList, start=1):
        print(img + " [" + str(i) + "/" + str(len(imgList)) + "]")
        imgOut = outDir + img
        img = inDir + img
        data = image.imread(img)

        # Rezise image for decreased computation time and improved performance
        # Commenting this out will likely require different threshold parameters
        data = util.resize_image(data, resize)
        h, w, _ = data.shape
        minSize = {"h" : h / 5, "w": w / 5}
        symmetries = []
        recursiveSym(data, symmetries, -1, minSize, rc)
        images.append([symmetries, data, imgOut])

    # Process images
    print("Processing symmetries ...")
    for i, img in enumerate(images, start=1):
        print("[" + str(i) + "/" + str(len(imgList)) + "]")
        symmetries = img[0].copy()
        data = img[1]
        imgOut = img[2]
        
        symmetries = util.placeInOrder(symmetries)
        if len(symmetries) != 0:
            # Slow mode uses the machine learning model, will increase performance for detecting rotational symmetries
            if args.mode == "slow":
                symmetries = util.removeBadSymmetries(symmetries, symThresholdBC, normThresholdBC)
                rotations = util.rotationalSymmetriesML(symmetries, model, data)
                symmetries = util.removeBadSymmetries(symmetries, symThresholdAC, normThresholdAC)
            else:
                symmetries = util.removeBadSymmetries(symmetries, symThresholdAC, normThresholdAC)
                rotations = util.rotationalSymmetries(symmetries, data, circleSymThreshold)

            util.removeSimilarLines(symmetries, data, lineSimilarity)
            util.removeSimilarRotational(rotations, data, rotationSimilarity)
            util.plotLines(symmetries)
            util.plotRotations(rotations)
        plt.imshow(data)    
        plt.savefig(imgOut[0:-4] + '.png')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        # print(symmetries)