import util
import numpy as np

def preproccesData(data):
    def isIntersecting(data):
        if util.line_intersect(data['line1x1'],data['line1y1'],data['line1x2'],data['line1y2'],data['line2x1'],data['line2y1'],data['line2x2'],data['line2y2']) == None:
            return False
        return True

    def distToIntersect1(data):
        intersect = util.line_intersect(data['line1x1'],data['line1y1'],data['line1x2'],data['line1y2'],data['line2x1'],data['line2y1'],data['line2x2'],data['line2y2'])
        if intersect == None:
            return -1
        else:
            Len = data['line1Len']
            subLen = np.sqrt( (data['line1x1'] - intersect[0])**2 + (data['line1y1'] - intersect[1])**2 )
            surface = data['width'] * data['height']
            return (abs(Len - (subLen * 2)) / surface)
    def distToIntersect2(data):
        intersect = util.line_intersect(data['line1x1'],data['line1y1'],data['line1x2'],data['line1y2'],data['line2x1'],data['line2y1'],data['line2x2'],data['line2y2'])
        if intersect == None:
            return -1
        else:
            Len = data['line1Len']
            subLen = np.sqrt( (data['line1x2'] - intersect[0])**2 + (data['line1y2'] - intersect[1])**2 )
            surface = data['width'] * data['height']
            return (abs(Len - (subLen * 2)) / surface)
    def distToIntersect3(data):
        intersect = util.line_intersect(data['line1x1'],data['line1y1'],data['line1x2'],data['line1y2'],data['line2x1'],data['line2y1'],data['line2x2'],data['line2y2'])
        if intersect == None:
            return -1
        else:
            Len = data['line1Len']
            subLen = np.sqrt( (data['line2x1'] - intersect[0])**2 + (data['line2y1'] - intersect[1])**2 )
            surface = data['width'] * data['height']
            return (abs(Len - (subLen * 2)) / surface)
    def distToIntersect4(data):
        intersect = util.line_intersect(data['line1x1'],data['line1y1'],data['line1x2'],data['line1y2'],data['line2x1'],data['line2y1'],data['line2x2'],data['line2y2'])
        if intersect == None:
            return -1
        else:
            Len = data['line1Len']
            subLen = np.sqrt( (data['line2x2'] - intersect[0])**2 + (data['line2y2'] - intersect[1])**2 )
            surface = data['width'] * data['height']
            return (abs(Len - (subLen * 2)) / surface)

    def calcPerpendicular(data):
        if data['line1Slope'] != 0:
            return (-1 * (1 / data['line1Slope']))
        elif data['line2Slope'] != 0:
            return (-1 * (1 / data['line2Slope']))
        else:
            return -1

    def calcPerpDiff(data):
        if data['line1Slope'] != 0:
            return (abs(data['linePerp'] - data['line2Slope']))
        elif data['line2Slope'] != 0:
            return (abs(data['linePerp'] - data['line1Slope']))
        else:
            return -1

    def calcSlope1(data):
        return util.getSlope([data['line1x1'],data['line1y1'],data['line1x2'],data['line1y2']], data['height'])

    def calcSlope2(data):
        return util.getSlope([data['line2x1'],data['line2y1'],data['line2x2'],data['line2y2']], data['height'])

    def meanToIntersect(data):
        return ((data['distToIntersect1'] + data['distToIntersect2'] + data['distToIntersect3'] + data['distToIntersect4']) / 4)


    #Score difference
    data = data.assign(ScoreDiff=abs(data['line1Score'] - data['line2Score']))

    #Slopes
    data['line1Slope'] = data.apply(lambda row: calcSlope1(row), axis=1)
    data['line2Slope'] = data.apply(lambda row: calcSlope2(row), axis=1)
    data = data.assign(SlopeDiff=abs(data['line1Slope'] - data['line2Slope']))

    #Perpendicular slopes
    data['linePerp'] = data.apply(lambda row: calcPerpendicular(row), axis=1)
    data['perpDiff'] = data.apply(lambda row: calcPerpDiff(row), axis=1)

    #Lengths
    data = data.assign(line1Len=np.sqrt( (data['line1x1'] - data['line1x2'])**2 + (data['line1y1'] - data['line1y2'])**2 ))
    data = data.assign(line2Len=np.sqrt( (data['line2x1'] - data['line2x2'])**2 + (data['line2y1'] - data['line2y2'])**2 ))
    data = data.assign(LenDiff=abs(data['line1Len'] - data['line2Len']))

    #Intersect
    data['intersect'] = data.apply(lambda row: isIntersecting(row), axis=1)
    data['distToIntersect1'] = data.apply(lambda row: distToIntersect1(row), axis=1)
    data['distToIntersect2'] = data.apply(lambda row: distToIntersect2(row), axis=1)
    data['distToIntersect3'] = data.apply(lambda row: distToIntersect3(row), axis=1)
    data['distToIntersect4'] = data.apply(lambda row: distToIntersect4(row), axis=1)

    #new
    data['meanDistToIntersect'] = data.apply(lambda row: meanToIntersect(row), axis=1)
    # data = data.assign(meanDistToIntersect=np.mean(data['distToIntersect1'], data['distToIntersect2'], data['distToIntersect3'], data['distToIntersect4']))
    data = data.assign(distToIntersectMean1=abs(data['distToIntersect1'] - data['meanDistToIntersect']))
    data = data.assign(distToIntersectMean2=abs(data['distToIntersect2'] - data['meanDistToIntersect']))
    data = data.assign(distToIntersectMean3=abs(data['distToIntersect3'] - data['meanDistToIntersect']))
    data = data.assign(distToIntersectMean4=abs(data['distToIntersect4'] - data['meanDistToIntersect']))

    #dropping inrelevant
    data = data.drop(['width', 'height', 'line1Score', 'line2Score'], axis=1)
    data = data.drop(['line1x1', 'line1y1', 'line1x2', 'line1y2'], axis=1)
    data = data.drop(['line2x1', 'line2y1', 'line2x2', 'line2y2'], axis=1)
    data = data.drop(['line1Len', 'line2Len'], axis=1)

    #new
    data = data.drop(['distToIntersect1', 'distToIntersect2', 'distToIntersect3', 'distToIntersect4', 'meanDistToIntersect'], axis=1)
    
    return data